import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset
from torch.autograd import Variable
from torchvision import transforms
import PIL
from models.bayes import _Bayes, BayesConv2d
from torch import distributions as dist
import torch.nn.functional as F
import torchvision
import os
import sys
import pickle
from sklearn.model_selection import train_test_split
from scipy.special import logsumexp
import yaml
import models.vae as vae_mod
import models
import pickle

DATA_ROOT = os.environ['DATA_ROOT']


def kl_ffn(mu0, var0, mu1, var1):
    return 0.5 * (var0/var1 + (mu1 - mu0)**2 / var1 - 1 + np.log(var1/var0))


def kl_term(net, backward=False, weight=1.):
    kl = 0.
    for m in net.modules():
        if isinstance(m, _Bayes):
            kl += m.klf(backward=backward, weight=weight)

    return kl


def kl_normal(m, backward=False, weight=1.):
    prior = m.prior
    q = m.dist()
    kl = weight * dist.kl_divergence(q, prior).sum()
    if backward:
        kl.backward()

    return kl.item()


def kl_loguniform_with_trunc_alpha(m, backward=False, weight=1.):
    c1 = 1.16145124
    c2 = -1.50204118
    c3 = 0.58629921

    alpha = m.get_alpha()
    log_alpha = -torch.log1p(torch.exp(-m.w_alpha))
    kl = -0.5 * log_alpha - c1 * alpha - c2 * (alpha**2) - c3 * (alpha**3)
    kl = kl.sum() * weight

    if backward:
        kl.backward()

    return kl.item()


def kl_loguniform(m, backward=False, weight=1.):
    k1, k2, k3 = 0.63576, 1.87320, 1.48695
    mu, std = m.q_params()
    log_alpha = 2 * (torch.log(std) - torch.log(torch.abs(mu) + 1e-16))

    kl = -k1 * torch.sigmoid(k2 + k3 * log_alpha) + 0.5 * torch.log1p(torch.exp(-log_alpha))
    kl = kl.sum() * weight

    if backward:
        kl.backward()

    return kl.item()


def kl_normal_mc(m, backward=False, weight=1., n_samples=5):
    prior = m.prior
    q = m.dist()
    kl = -q.entropy().sum()
    for _ in range(n_samples):
        kl -= prior.log_prob(q.rsample()).sum() / float(n_samples)
    if backward:
        kl.backward()

    return kl.item()


def kl_dwp(vae, n_tries=1, normalize=False):
    vae = vae if isinstance(vae, models.vae.VAE) else vae.module
    z_prior = dist.Normal(torch.FloatTensor([0]).to(vae.device),
                          torch.FloatTensor([1]).to(vae.device))

    def foo(m, backward=False, weight=1.):
        q = m.dist()
        qent = -q.entropy().sum() * weight
        if backward:
            qent.backward()
        kl = qent.item()
        mean, scale = m.q_params()
        N = mean.shape[0] * mean.shape[1]
        K = mean.shape[-1]
        BS = 1000000

        for i in range(0, N, BS):
            loss = 0
            for _ in range(n_tries):
                mean, scale = m.q_params()
                w = dist.Normal(mean.view((-1, 1, K, K))[i:i + BS], scale.view((-1, 1, K, K))[i:i + BS]).rsample()
                if normalize:
                    norm = torch.sqrt((w**2).sum(2).sum(2)).view((-1, 1, 1, 1))
                    w = w / (norm + 1e-6)

                # z_mu, z_var = vae.encode(w)
                # w_mu, w_var = vae.decode(z_mu)
                (z_mu, z_var), (w_mu, w_var) = vae(w)
                logp_theta = dist.Normal(w_mu, torch.sqrt(w_var)).log_prob(w).sum()
                z_posterior = dist.Normal(z_mu, torch.sqrt(z_var))
                kl_z = dist.kl_divergence(z_posterior, z_prior).sum()
                loss += weight * (-logp_theta + kl_z) / float(n_tries)

            kl += loss.item()
            if backward:
                loss.backward()

        return kl

    return foo


def kl_flow(flow, n_tries=1):
    def foo(m, backward=False, weight=1.):
        q = m.dist()
        qent = -q.entropy().sum() * weight
        if backward:
            qent.backward()

        mean, scale = m.q_params()
        N = mean.shape[0] * mean.shape[1]
        K = mean.shape[-1]
        kl = qent.item()
        loss = 0.
        for _ in range(n_tries):
            mean, scale = m.q_params()
            w = dist.Normal(mean.view((-1, K*K)), scale.view((-1, K*K))).rsample()
            loss -= weight * flow.log_prob(w).sum() / float(n_tries)

        if backward:
            loss.backward()

        kl += loss.item()
        return kl

    return foo


def dwp_regularizer(vae, m, n_tries=1, backward=False, weight=1., target='elbo'):
    # assert n_tries == 1
    z_prior = dist.Normal(torch.FloatTensor([0]).to(vae.device),
                          torch.FloatTensor([1]).to(vae.device))
    w = m.mean.weight if isinstance(m, _Bayes) else m.weight
    K = w.shape[-1]
    w = w.view((-1, 1, K, K))

    if target == 'elbo':
        (z_mu, z_var), (w_mu, w_var) = vae(w)
        loss = -dist.Normal(w_mu, torch.sqrt(w_var)).log_prob(w).sum() / float(n_tries)
        loss += dist.kl_divergence(dist.Normal(z_mu, torch.sqrt(z_var)), z_prior).sum()

        for _ in range(n_tries - 1):
            z = dist.Normal(z_mu, z_var).rsample()
            w_mu, w_var = vae.decode(z)
            loss = -dist.Normal(w_mu, torch.sqrt(w_var)).log_prob(w).sum() / float(n_tries)
    elif target == 'logsumexp':
        z_mu, z_var = vae.encode(w)
        loss = 0.
        z = dist.Normal(z_mu, z_var).rsample((n_tries,))
        w_mu, w_var = vae.decode(z)
        logit = dist.Normal(w_mu, torch.sqrt(w_var)).log_prob(w).sum(dim=(1, 2, 3, 4))
        logit += z_prior.log_prob(z).sum(dim=(1, 2, 3, 4))
        logit -= dist.Normal(z_mu, z_var).log_prob(z).sum(dim=(1, 2, 3, 4))
        logit -= np.log(n_tries)
        loss = torch.logsumexp(logit)
        loss *= -1.
    else:
        raise NotImplementedError

    if backward:
        (loss * weight).backward()

    return loss.item()

    # BS = 10000000

    # for i in range(0, N, BS):
    #     loss = 0.
    #     w_batch = w.view((-1, 1, K, K)[i: i + BS])
    #     for s in range(n_tries):
    #         (z_mu, z_var), (w_mu, w_var) = vae(w_batch)
    #         if s == 0:
    #             prior += weight * dist.kl_divergence(dist.Normal(z_mu, torch.sqrt(z_var)), z_prior).sum().item()
    #         loss -= weight * dist.Normal(w_mu, w_var).log_prob(w_batch).sum() / float(n_tries)

    #     prior += loss.item()
    #     if backward:
    #         loss.backward()

    # return prior


class ConvDataset(Dataset):
    def __init__(self, file=None, data=None):
        super(ConvDataset, self).__init__()
        if file is not None:
            self.data = np.load(file)
        elif data is not None:
            self.data = np.copy(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx])


def get_dataloaders(file, train_bs, test_bs, train_size=0.8):
    train, test = train_test_split(np.load(file), train_size=train_size)
    D = train.shape[-1]
    train, test = ConvDataset(data=train), ConvDataset(data=test)
    trainloader = torch.utils.data.DataLoader(train, batch_size=train_bs)
    testloader = torch.utils.data.DataLoader(test, batch_size=test_bs)

    return trainloader, testloader, D


def load_dataset(data, train_bs, test_bs, num_examples=None, augmentation=True, data_root=DATA_ROOT,
                 shuffle=True, seed=42):
    transform_train = transforms.Compose([
        MyPad(4),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    if data == 'cifar':
        trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True,
                                                transform=transform_train if augmentation else transform_test)
        testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)
        if num_examples is not None and num_examples != len(trainset):
            a, _, b, _ = train_test_split(trainset.train_data, trainset.train_labels,
                                          train_size=num_examples, random_state=42)
            trainset.train_data = a
            trainset.train_labels = b
    elif data == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root=data_root, train=True, download=True,
                                                 transform=transform_train if augmentation else transform_test)
        testset = torchvision.datasets.CIFAR100(root=data_root, train=False, download=True, transform=transform_test)
        if num_examples is not None and num_examples != len(trainset):
            a, _, b, _ = train_test_split(trainset.train_data, trainset.train_labels,
                                          train_size=num_examples, random_state=42)
            trainset.train_data = a
            trainset.train_labels = b
    elif data == 'svhn':
        trainset = torchvision.datasets.SVHN(root=data_root, split='train', download=True,
                                             transform=transform_train if augmentation else transform_test)
        testset = torchvision.datasets.SVHN(root=data_root, split='test', download=True, transform=transform_test)
        if num_examples is not None and num_examples != len(trainset):
            a, _, b, _ = train_test_split(trainset.data, trainset.labels,
                                          train_size=num_examples, random_state=42)
            trainset.data = a
            trainset.labels = b
    elif data == 'mnist':
        trainset = torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform=transform_test)
        testset = torchvision.datasets.MNIST(root=data_root, train=False, download=True, transform=transform_test)

        if num_examples is not None and num_examples != len(trainset):
            idxs, _ = train_test_split(np.arange(len(trainset)), train_size=num_examples, random_state=seed,
                                       stratify=tonp(trainset.train_labels))
            trainset.train_data = trainset.train_data[idxs]
            trainset.train_labels = trainset.train_labels[idxs]
    elif data == 'cifar5':
        CIFAR5_CLASSES = [0, 1, 2, 3, 4]
        trainset = CIFAR(root=data_root, train=True, download=True,
                         transform=transform_train if augmentation else transform_test, classes=CIFAR5_CLASSES,
                         random_labeling=False)
        testset = CIFAR(root=data_root, train=False, download=True, transform=transform_test, classes=CIFAR5_CLASSES,
                        random_labeling=False)
    elif data == 'not-mnist':
        trainset = torchvision.datasets.MNIST(root=os.path.join(data_root, 'not-mnist'), train=True,
                                              download=True, transform=transform_test)
        testset = torchvision.datasets.MNIST(root=os.path.join(data_root, 'not-mnist'), train=False,
                                             download=True, transform=transform_test)
        if num_examples is not None and num_examples != len(trainset):
            idxs, _ = train_test_split(np.arange(len(trainset)), train_size=num_examples, random_state=seed,
                                       stratify=tonp(trainset.train_labels))
            trainset.train_data = trainset.train_data[idxs]
            trainset.train_labels = trainset.train_labels[idxs]
    elif data == 'cifar5-rest':
        CIFAR5_CLASSES = [5, 6, 7, 8, 9]
        trainset = CIFAR(root=data_root, train=True, download=True,
                         transform=transform_train if augmentation else transform_test, classes=CIFAR5_CLASSES)
        testset = CIFAR(root=data_root, train=False, download=True, transform=transform_test, classes=CIFAR5_CLASSES)
    elif data == 'shapes':
        train_images = np.load(os.path.join(data_root, 'four-shapes/dataset/train_images.npy'))
        test_images = np.load(os.path.join(data_root, 'four-shapes/dataset/test_images.npy'))
        train_labels = np.load(os.path.join(data_root, 'four-shapes/dataset/train_labels.npy'))
        test_labels = np.load(os.path.join(data_root, 'four-shapes/dataset/test_labels.npy'))

        if num_examples != 4000:
            RuntimeWarning('==> --num-examples for shapes dataset should be 4000 <==')

        if num_examples is not None:
            train_images, _, train_labels, _ = train_test_split(train_images, train_labels, train_size=num_examples,
                                                                random_state=seed, stratify=train_labels)

        train_images, test_images = map(torch.Tensor, [train_images, test_images])
        train_labels, test_labels = map(torch.LongTensor, [train_labels, test_labels])

        trainset = torch.utils.data.TensorDataset(train_images, train_labels)
        testset = torch.utils.data.TensorDataset(test_images, test_labels)
    else:
        raise NotImplementedError

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_bs, shuffle=shuffle, num_workers=0)

    testloader = torch.utils.data.DataLoader(testset, batch_size=test_bs, shuffle=False, num_workers=0)

    return trainloader, testloader


def pad(img, size, mode):
    if isinstance(img, PIL.Image.Image):
        img = np.array(img)
    return np.pad(img, [(size, size), (size, size), (0, 0)], mode)


class MyPad(object):
    def __init__(self, size, mode='reflect'):
        self.mode = mode
        self.size = size
        self.topil = transforms.ToPILImage()

    def __call__(self, img):
        return self.topil(pad(img, self.size, self.mode))


def get_dataloader(file, bs, shuffle=False):
    data = np.load(file)
    D = data.shape[-1]
    data = ConvDataset(data=data)
    dataloader = torch.utils.data.DataLoader(data, batch_size=bs, shuffle=shuffle)
    return dataloader, D


class FFGKL(nn.Module):
    """KL divergence between standart normal prior and fully-factorize gaussian posterior"""

    def __init__(self):
        super(FFGKL, self).__init__()

    def forward(self, mu, var):
        return -0.5 * (1 + torch.log(var) - mu.pow(2) - var).sum()


class VAEELBOLoss(nn.Module):
    """docstring for ELBOLoss"""

    def __init__(self, likelihood=None, kl=None, use_cuda=False):
        super(VAEELBOLoss, self).__init__()
        if likelihood is None:
            self.likelihood = NormalLikelihood()
        if kl is None:
            self.kl = FFGKL()
        if use_cuda:
            self.likelihood = self.likelihood.cuda()
            self.kl = self.kl.cuda()

    def forward(self, target, likelihood_params, var_params):
        return self.likelihood(target, *likelihood_params), self.kl(*var_params)


class NormalLikelihood(nn.Module):
    def __init__(self):
        super(NormalLikelihood, self).__init__()

    def forward(self, target, mu, var):
        loss = torch.sum(-(target - mu)**2 / var - np.log(2 * np.pi) - torch.log(var)) * 0.5
        return loss


def tonp(x):
    return x.cpu().detach().numpy()


def get_kernels(net):
    convs = [m for m in net.modules() if isinstance(m, nn.Conv2d)]
    weights = [tonp(m.weight) for m in convs]
    weights = [w.reshape((-1, w.shape[-2], w.shape[-1])) for w in weights]
    return weights


def load_vae(path, device=None):
    with open(os.path.join(path, 'params.yaml')) as f:
        vae_args = yaml.load(f)

    if vae_args['kernel_dim'] == 5:
        decoder = vae_mod.Decoder5x5(vae_args['z_dim'], vae_args['hidden_dim'], var=vae_args['var'])
        encoder = vae_mod.Encoder5x5(vae_args['z_dim'], vae_args['hidden_dim'])
    if vae_args['kernel_dim'] == 7:
        decoder = vae_mod.Decoder7x7(vae_args['z_dim'], vae_args['hidden_dim'], var=vae_args['var'])
        encoder = vae_mod.Encoder7x7(vae_args['z_dim'], vae_args['hidden_dim'])
    elif vae_args['kernel_dim'] == 16:
        decoder = vae_mod.Decoder16x16(vae_args['z_dim'], vae_args['hidden_dim'], var=vae_args['var'])
        encoder = vae_mod.Encoder16x16(vae_args['z_dim'], vae_args['hidden_dim'])

    vae = vae_mod.VAE(encoder, decoder, device=device)
    vae.load_state_dict(torch.load(os.path.join(path, 'vae_params.torch')))

    return vae


def load_flow(path, device):
    with open(os.path.join(path, 'params.yaml'), 'rb') as f:
        params = yaml.load(f)

    if params['kernel_dim'] == 5:
        nets, nett = models.flows.snet5x5, models.flows.tnet5x5
    elif params['kernel_dim'] == 7:
        nets, nett = models.flows.snet7x7, models.flows.tnet7x7
    else:
        raise NotImplementedError

    k2 = int(params['kernel_dim'] ** 2)
    masks = torch.from_numpy((np.random.uniform(0, 1, size=(16, k2)) < 0.05).astype(np.float32))
    prior = dist.MultivariateNormal(torch.zeros(k2).to(device), torch.eye(k2).to(device))
    flow = models.flows.RealNVP(nets, nett, masks, prior, device=device)
    flow.load_state_dict(pickle.load(open(os.path.join(path, 'flow_params.torch'), 'rb')))

    return flow


def net_init(net, init, vae_path=None):
    if hasattr(net, 'weights_init'):
        net.weights_init(init, vae_path)
        return

    if init == 'xavier':
        net.apply(weight_init(module=nn.Conv2d, initf=nn.init.xavier_normal_))
        net.apply(weight_init(module=nn.Linear, initf=nn.init.xavier_normal_))
    elif init == 'vae':
        vae = load_vae(vae_path, net.device)
        net.apply(weight_init(module=nn.Conv2d, initf=vae_init(vae)))
    elif init == 'flow':
        flow = load_flow(vae_path, net.device)
        net.apply(weight_init(module=nn.Conv2d, initf=flow_init(flow)))
    else:
        raise NotImplementedError('No init method "{}"'.format(init))

    net.apply(weight_init(module=models.bayes.LogScaleConv2d, initf=const_init(-10.)))
    net.apply(weight_init(module=models.bayes.LogScaleLinear, initf=const_init(-10.)))


def weight_init(module, initf):
    def foo(m):
        classname = m.__class__.__name__.lower()
        if isinstance(m, module):
            initf(m.weight)
    return foo


def const_init(val):
    def foo(var):
        nn.init.constant_(var, val)
    return foo


def vae_init(vae):
    def foo(var):
        shape = var.shape
        z = torch.randn(shape[0] * shape[1], vae.encoder.z_dim, 1, 1).to(vae.device)
        w, _ = vae.decode(z)
        var.data = w.detach().view(*shape)
    return foo


def flow_init(flow):
    def foo(var):
        shape = var.shape
        k = var.shape[-1]
        w = flow.sample(shape[0] * shape[1]).view(shape[0] * shape[1], 1, k, k)
        var.data = w.detach().view(*shape)
    return foo


def pretrained_init(net):
    weights = torch.cat([m.weight.view((-1, 3, 3)).detach() for m in net.modules() if isinstance(m, nn.Conv2d)])
    N = weights.shape[0]

    def foo(var):
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(var)
        std = np.sqrt(2.0 / (fan_in + fan_out))
        M = np.prod(var.shape[:2])
        idxs = np.random.choice(N, size=M, replace=True)
        w_std = torch.std(weights[idxs], unbiased=True).to(var.device)
        var.data = weights[idxs].view(*var.shape).to(var.device) * std / w_std

    return foo


class MovingMetric(object):
    def __init__(self):
        self.n = 0.
        self.val = 0.

    def add(self, v, n):
        self.n += n
        self.val += v

    def get_val(self):
        return self.val / max(self.n, 1)


def mc_ensemble(net, dataloader, n_tries=10, log=False):
    gt = []
    pred = []
    for x, y in dataloader:
        x = x.to(net.device)
        gt.append(y.numpy())
        ens = None
        for i in range(n_tries):
            p = tonp(F.log_softmax(net(x), dim=1))
            if ens is None:
                ens = p
            else:
                w = np.array([1 if i > 1 else (1. / n_tries), 1./n_tries])[:, np.newaxis, np.newaxis]
                ens = logsumexp(np.stack([ens, p]), axis=0, b=w)
        pred.append(ens)

    if log:
        return np.concatenate(pred), np.concatenate(gt)

    return np.exp(np.concatenate(pred)), np.concatenate(gt)


def get_logp(net, dataloader):
    gt = []
    logits = []
    for x, y in dataloader:
        x = x.to(net.device)
        gt.append(y.numpy())
        p = tonp(F.log_softmax(net(x)))
        logits.append(p)

    return np.concatenate(logits), np.concatenate(gt)


def bnn_mode(net, mode):
    if mode not in ['det', 'stoch']:
        raise NotImplementedError

    for m in net.modules():
        if isinstance(m, _Bayes):
            m._mode = mode


class CIFAR(torchvision.datasets.CIFAR10):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset with several classes.
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False, classes=None, random_labeling=False):

        if classes is None:
            classes = np.arange(10).tolist()

        self.classes = classes[:]

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.random_labeling = random_labeling

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                else:
                    self.train_labels += entry['fine_labels']
                fo.close()

            mask = np.isin(self.train_labels, classes)
            self.train_labels = [classes.index(l) for l, cond in zip(self.train_labels, mask) if cond]
            if self.random_labeling:
                self.train_labels = np.random.permutation(self.train_labels)

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))[mask]
            self.train_data = self.train_data.transpose((0, 2, 3, 1))
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()

            mask = np.isin(self.test_labels, classes)
            self.test_labels = [classes.index(l) for l, cond in zip(self.test_labels, mask) if cond]

            self.test_data = self.test_data.reshape((10000, 3, 32, 32))[mask]
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC
