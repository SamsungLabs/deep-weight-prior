import torch
from models import bayes
from torch import nn
from collections import OrderedDict
import utils
from torch import distributions as dist
import numpy as np


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CIFARNet(bayes._BayesNet):
    def __init__(self, cfg, device=None, n_classes=10, do=[], k=1., vae_list=None, **kwargs):
        super(CIFARNet, self).__init__(**kwargs)
        self.device = device
        self.cfg = cfg

        d1, d2, d3 = map(int, [128 * k, 256 * k, 512 * k])

        if cfg == 'vanilla':
            # 3x32x32
            self.features = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(3, d1, 7)),  # 128x26x26
                ('bn1', nn.BatchNorm2d(d1)),
                ('relu1', nn.LeakyReLU()),
                ('maxpool', nn.MaxPool2d(2)),  # 128x13x13

                ('conv2', nn.Conv2d(d1, d2, 5)),  # 256x9x9
                ('bn2', nn.BatchNorm2d(d2)),
                ('relu2', nn.LeakyReLU()),

                ('conv3', nn.Conv2d(d2, d2, 5)),  # 256x5x5
                ('bn3', nn.BatchNorm2d(d2)),
                ('relu3', nn.LeakyReLU()),

                ('conv4', nn.Conv2d(d2, 512, 5)),  # 512x1x1
                ('bn4', nn.BatchNorm2d(512)),
                ('relu4', nn.LeakyReLU()),

                ('flatten', Flatten()),  # 512
            ]))
        elif cfg == 'bayes':
            # 3x32x32
            self.features = nn.Sequential(OrderedDict([
                ('conv1', bayes.BayesConv2d(3, d1, 7)),  # 128x26x26
                ('bn1', nn.BatchNorm2d(d1)),
                ('relu1', nn.LeakyReLU()),
                ('maxpool', nn.MaxPool2d(2)),  # 128x13x13

                ('conv2', bayes.BayesConv2d(d1, d2, 5)),  # 256x9x9
                ('bn2', nn.BatchNorm2d(d2)),
                ('relu2', nn.LeakyReLU()),

                ('conv3', bayes.BayesConv2d(d2, d2, 5)),  # 256x5x5
                ('bn3', nn.BatchNorm2d(d2)),
                ('relu3', nn.LeakyReLU()),

                ('conv4', bayes.BayesConv2d(d2, 512, 5)),  # 512x1x1
                ('bn4', nn.BatchNorm2d(512)),
                ('relu4', nn.LeakyReLU()),

                ('flatten', Flatten()),  # 512
            ]))
        elif cfg == 'bayes1110':
            # 3x32x32
            self.features = nn.Sequential(OrderedDict([
                ('conv1', bayes.BayesConv2d(3, d1, 7)),  # 128x26x26
                ('bn1', nn.BatchNorm2d(d1)),
                ('relu1', nn.LeakyReLU()),
                ('maxpool', nn.MaxPool2d(2)),  # 128x13x13

                ('conv2', bayes.BayesConv2d(d1, d2, 5)),  # 256x9x9
                ('bn2', nn.BatchNorm2d(d2)),
                ('relu2', nn.LeakyReLU()),

                ('conv3', bayes.BayesConv2d(d2, d2, 5)),  # 256x5x5
                ('bn3', nn.BatchNorm2d(d2)),
                ('relu3', nn.LeakyReLU()),

                ('conv4', nn.Conv2d(d2, 512, 5)),  # 512x1x1
                ('bn4', nn.BatchNorm2d(512)),
                ('relu4', nn.LeakyReLU()),

                ('flatten', Flatten()),  # 512
            ]))
        elif cfg == 'bayes1100':
            # 3x32x32
            self.features = nn.Sequential(OrderedDict([
                ('conv1', bayes.BayesConv2d(3, d1, 7)),  # 128x26x26
                ('bn1', nn.BatchNorm2d(d1)),
                ('relu1', nn.LeakyReLU()),
                ('maxpool', nn.MaxPool2d(2)),  # 128x13x13

                ('conv2', bayes.BayesConv2d(d1, d2, 5)),  # 256x9x9
                ('bn2', nn.BatchNorm2d(d2)),
                ('relu2', nn.LeakyReLU()),

                ('conv3', nn.Conv2d(d2, d2, 5)),  # 256x5x5
                ('bn3', nn.BatchNorm2d(d2)),
                ('relu3', nn.LeakyReLU()),

                ('conv4', nn.Conv2d(d2, 512, 5)),  # 512x1x1
                ('bn4', nn.BatchNorm2d(512)),
                ('relu4', nn.LeakyReLU()),

                ('flatten', Flatten()),  # 512
            ]))
        elif cfg == 'bayes1000':
            # 3x32x32
            self.features = nn.Sequential(OrderedDict([
                ('conv1', bayes.BayesConv2d(3, d1, 7)),  # 128x26x26
                ('bn1', nn.BatchNorm2d(d1)),
                ('relu1', nn.LeakyReLU()),
                ('maxpool', nn.MaxPool2d(2)),  # 128x13x13

                ('conv2', nn.Conv2d(d1, d2, 5)),  # 256x9x9
                ('bn2', nn.BatchNorm2d(d2)),
                ('relu2', nn.LeakyReLU()),

                ('conv3', nn.Conv2d(d2, d2, 5)),  # 256x5x5
                ('bn3', nn.BatchNorm2d(d2)),
                ('relu3', nn.LeakyReLU()),

                ('conv4', nn.Conv2d(d2, 512, 5)),  # 512x1x1
                ('bn4', nn.BatchNorm2d(512)),
                ('relu4', nn.LeakyReLU()),

                ('flatten', Flatten()),  # 512
            ]))
        elif cfg == 'bayes-mtrunca':
            # 3x32x32
            self.features = nn.Sequential(OrderedDict([
                ('conv1', bayes.MuTruncAlphaFFGConv2d(3, d1, 7)),  # 128x26x26
                ('bn1', nn.BatchNorm2d(d1)),
                ('relu1', nn.LeakyReLU()),
                ('maxpool', nn.MaxPool2d(2)),  # 128x13x13

                ('conv2', bayes.MuTruncAlphaFFGConv2d(d1, d2, 5)),  # 256x9x9
                ('bn2', nn.BatchNorm2d(d2)),
                ('relu2', nn.LeakyReLU()),

                ('conv3', bayes.MuTruncAlphaFFGConv2d(d2, d2, 5)),  # 256x5x5
                ('bn3', nn.BatchNorm2d(d2)),
                ('relu3', nn.LeakyReLU()),

                ('conv4', bayes.MuTruncAlphaFFGConv2d(d2, 512, 5)),  # 512x1x1
                ('bn4', nn.BatchNorm2d(512)),
                ('relu4', nn.LeakyReLU()),

                ('flatten', Flatten()),  # 512
            ]))
        elif cfg == 'bayes1100-mtrunca':
            # 3x32x32
            self.features = nn.Sequential(OrderedDict([
                ('conv1', bayes.MuTruncAlphaFFGConv2d(3, d1, 7)),  # 128x26x26
                ('bn1', nn.BatchNorm2d(d1)),
                ('relu1', nn.LeakyReLU()),
                ('maxpool', nn.MaxPool2d(2)),  # 128x13x13

                ('conv2', bayes.MuTruncAlphaFFGConv2d(d1, d2, 5)),  # 256x9x9
                ('bn2', nn.BatchNorm2d(d2)),
                ('relu2', nn.LeakyReLU()),

                ('conv3', nn.Conv2d(d2, d2, 5)),  # 256x5x5
                ('bn3', nn.BatchNorm2d(d2)),
                ('relu3', nn.LeakyReLU()),

                ('conv4', nn.Conv2d(d2, 512, 5)),  # 512x1x1
                ('bn4', nn.BatchNorm2d(512)),
                ('relu4', nn.LeakyReLU()),

                ('flatten', Flatten()),  # 512
            ]))
        elif cfg == 'bayes1000-mtrunca':
            # 3x32x32
            self.features = nn.Sequential(OrderedDict([
                ('conv1', bayes.MuTruncAlphaFFGConv2d(3, d1, 7)),  # 128x26x26
                ('bn1', nn.BatchNorm2d(d1)),
                ('relu1', nn.LeakyReLU()),
                ('maxpool', nn.MaxPool2d(2)),  # 128x13x13

                ('conv2', nn.Conv2d(d1, d2, 5)),  # 256x9x9
                ('bn2', nn.BatchNorm2d(d2)),
                ('relu2', nn.LeakyReLU()),

                ('conv3', nn.Conv2d(d2, d2, 5)),  # 256x5x5
                ('bn3', nn.BatchNorm2d(d2)),
                ('relu3', nn.LeakyReLU()),

                ('conv4', nn.Conv2d(d2, 512, 5)),  # 512x1x1
                ('bn4', nn.BatchNorm2d(512)),
                ('relu4', nn.LeakyReLU()),

                ('flatten', Flatten()),  # 512
            ]))
        elif cfg == 'vanilla-do':
            # 3x32x32
            self.features = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(3, d1, 7)),  # 128x26x26
                ('bn1', nn.BatchNorm2d(d1)),
                ('relu1', nn.LeakyReLU()),
                ('maxpool', nn.MaxPool2d(2)),  # 128x13x13

                ('conv2', nn.Conv2d(d1, d2, 5)),  # 256x9x9
                ('bn2', nn.BatchNorm2d(d2)),
                ('relu2', nn.LeakyReLU()),

                ('conv3', nn.Conv2d(d2, d2, 5)),  # 256x5x5
                ('bn3', nn.BatchNorm2d(d2)),
                ('relu3', nn.LeakyReLU()),

                ('conv4', nn.Conv2d(d2, 512, 5)),  # 512x1x1
                ('bn4', nn.BatchNorm2d(512)),
                ('relu4', nn.LeakyReLU()),

                ('flatten', Flatten()),  # 512
            ]))
        else:
            raise NotImplementedError

        self.classifier = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(512, 512)),
                ('bn1', nn.BatchNorm1d(512)),
                ('relu1', nn.LeakyReLU()),
                ('linear', nn.Linear(512, n_classes))
        ]))

        if self.device:
            self.to(self.device)

    def forward(self, input):
        return self.classifier(self.features(input))

    def weights_init(self, init_list, vae_list, flow_list=None, pretrained=None, filters_list=None):
        self.apply(utils.weight_init(module=nn.Conv2d, initf=nn.init.xavier_normal_))
        self.apply(utils.weight_init(module=nn.Linear, initf=nn.init.xavier_normal_))
        self.apply(utils.weight_init(module=bayes.LogScaleConv2d, initf=utils.const_init(-10.)))
        self.apply(utils.weight_init(module=bayes.LogScaleLinear, initf=utils.const_init(-10.)))

        if len(init_list) > 0 and init_list[0] == 'pretrained':
            assert len(init_list) == 1
            w_pretrained = torch.load(pretrained)
            for k, v in w_pretrained.items():
                if k in self.state_dict():
                    self.state_dict()[k].data.copy_(v)
                else:
                    tokens = k.split('.')
                    self.state_dict()['.'.join(tokens[:2] + ['mean'] + tokens[-1:])].data.copy_(v)
            return

        convs = [self.features.conv1, self.features.conv2, self.features.conv3, self.features.conv4]
        for i, m in enumerate(convs):
            init = init_list[i] if i < len(init_list) else 'xavier'
            w = m.mean.weight if isinstance(m, bayes._Bayes) else m.weight
            if init == 'vae':
                vae_path = vae_list[i]
                vae = utils.load_vae(vae_path, device=self.device)
                z = torch.randn(w.size(0) * w.size(1), vae.encoder.z_dim, 1, 1).to(vae.device)
                x = vae.decode(z)[0]
                w.data = x.reshape(w.shape)
            elif init == 'flow':
                flow_path = flow_list[i]
                flow = utils.load_flow(flow_path, device=self.device)
                utils.flow_init(flow)(w)
            elif init == 'xavier':
                pass
            elif init == 'filters':
                filters = np.load(filters_list[i])
                filters = np.concatenate([filters]*10)
                N = np.prod(w.shape[:2])
                filters = filters[np.random.permutation(len(filters))[:N]]
                w.data = torch.from_numpy(filters.reshape(*w.shape)).to(self.device)
            elif init == 'recon':
                filters = np.load(filters_list[i])
                filters = np.concatenate([filters]*10)
                N = np.prod(w.shape[:2])
                filters = filters[np.random.permutation(len(filters))[:N]]
                vae_path = vae_list[i]
                vae = utils.load_vae(vae_path, device=self.device)
                filters = vae(torch.from_numpy(filters).to(self.device))[1][0]
                w.data = filters.reshape_as(w)
            else:
                raise NotImplementedError('no {} init'.format(init))

    def set_prior(self, prior_list, dwp_samples, vae_list, flow_list=None):
        convs = [self.features.conv1, self.features.conv2, self.features.conv3, self.features.conv4]
        for i, m in enumerate(convs):
            if not isinstance(m, bayes._Bayes):
                continue

            if prior_list[i] == 'vae':
                vae = utils.load_vae(vae_list[i], self.device)
                vae = nn.DataParallel(vae)
                for p in vae.parameters():
                    p.requires_grad = False
                m.kl_function = utils.kl_dwp(vae, n_tries=dwp_samples)
            elif prior_list[i] == 'flow':
                flow = utils.load_flow(flow_list[i], self.device)
                for p in flow.parameters():
                    p.requires_grad = False
                m.kl_function = utils.kl_flow(flow, n_tries=dwp_samples)
            elif prior_list[i] == 'sn':
                m.kl_function = utils.kl_normal
                m.prior = dist.Normal(torch.FloatTensor([0.]).to(self.device),
                                      torch.FloatTensor([1.]).to(self.device))
            elif prior_list[i] == 'loguniform':
                if self.cfg in ['bayes-mtrunca', 'bayes1100-mtrunca', 'bayes1000-mtrunca']:
                    m.kl_function = utils.kl_loguniform_with_trunc_alpha
                else:
                    raise NotImplementedError
            elif prior_list[i] == 'no':
                pass
            else:
                raise NotImplementedError


class CIFARNetNew(bayes._BayesNet):
    def __init__(self, cfg, device=None, n_classes=10, do=[], k=1., vae_list=None,
                 logvar=-10., **kwargs):
        super(CIFARNetNew, self).__init__(**kwargs)
        self.device = device
        self.cfg = cfg
        self.vaes = []

        d1, d2, d3 = map(int, [128 * k, 256 * k, 512 * k])

        if cfg in ['vanilla', 'vanilla-nofc', 'vanilla-do']:
            # 3x32x32
            self.features = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(3, d1, 7)),  # 128x26x26
                ('relu1', nn.LeakyReLU()),
                ('maxpool', nn.MaxPool2d(2)),  # 128x13x13

                ('conv2', nn.Conv2d(d1, d2, 5)),  # 256x9x9
                ('relu2', nn.LeakyReLU()),

                ('conv3', nn.Conv2d(d2, d2, 5)),  # 256x5x5
                ('relu3', nn.LeakyReLU()),

                ('flatten', Flatten()),  # 512
            ]))
        elif cfg == 'bayes111' or cfg == 'bayes111-nofc':
            # 3x32x32
            self.features = nn.Sequential(OrderedDict([
                ('conv1', bayes.BayesConv2d(3, d1, 7)),  # 128x26x26
                ('relu1', nn.LeakyReLU()),
                ('maxpool', nn.MaxPool2d(2)),  # 128x13x13

                ('conv2', bayes.BayesConv2d(d1, d2, 5)),  # 256x9x9
                ('relu2', nn.LeakyReLU()),

                ('conv3', bayes.BayesConv2d(d2, d2, 5)),  # 256x5x5
                ('relu3', nn.LeakyReLU()),

                ('flatten', Flatten()),  # 512
            ]))
        elif cfg == 'bayes111-mutrunca':
            # 3x32x32
            self.features = nn.Sequential(OrderedDict([
                ('conv1', bayes.MuTruncAlphaFFGConv2d(3, d1, 7)),  # 128x26x26
                ('relu1', nn.LeakyReLU()),
                ('maxpool', nn.MaxPool2d(2)),  # 128x13x13

                ('conv2', bayes.MuTruncAlphaFFGConv2d(d1, d2, 5)),  # 256x9x9
                ('relu2', nn.LeakyReLU()),

                ('conv3', bayes.MuTruncAlphaFFGConv2d(d2, d2, 5)),  # 256x5x5
                ('relu3', nn.LeakyReLU()),

                ('flatten', Flatten()),  # 512
            ]))
        elif cfg == 'bayes110':
            # 3x32x32
            self.features = nn.Sequential(OrderedDict([
                ('conv1', bayes.BayesConv2d(3, d1, 7)),  # 128x26x26
                ('relu1', nn.LeakyReLU()),
                ('maxpool', nn.MaxPool2d(2)),  # 128x13x13

                ('conv2', bayes.BayesConv2d(d1, d2, 5)),  # 256x9x9
                ('relu2', nn.LeakyReLU()),

                ('conv3', nn.Conv2d(d2, d2, 5)),  # 256x5x5
                ('relu3', nn.LeakyReLU()),

                ('flatten', Flatten()),  # 512
            ]))
        elif cfg == 'bayes100' or cfg == 'bayes100-nofc':
            # 3x32x32
            self.features = nn.Sequential(OrderedDict([
                ('conv1', bayes.BayesConv2d(3, d1, 7)),  # 128x26x26
                ('relu1', nn.LeakyReLU()),
                ('maxpool', nn.MaxPool2d(2)),  # 128x13x13

                ('conv2', nn.Conv2d(d1, d2, 5)),  # 256x9x9
                ('relu2', nn.LeakyReLU()),

                ('conv3', nn.Conv2d(d2, d2, 5)),  # 256x5x5
                ('relu3', nn.LeakyReLU()),

                ('flatten', Flatten()),  # 512
            ]))
        else:
            raise NotImplementedError

        if 'nofc' in self.cfg:
            self.classifier = nn.Sequential(OrderedDict([
                ('linear', nn.Linear(d2 * 25, n_classes))
            ]))
            print('====> CIFARNetNew without FC!!!!!')
        elif 'do' in self.cfg:
            self.classifier = nn.Sequential(OrderedDict([
                ('do1', nn.Dropout(0.5)),
                ('fc1', nn.Linear(d2 * 25, 512)),
                ('relu1', nn.LeakyReLU()),
                ('do2', nn.Dropout(0.2)),
                ('linear', nn.Linear(512, n_classes))
            ]))
        else:
            self.classifier = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(d2 * 25, 512)),
                ('relu1', nn.LeakyReLU()),
                ('linear', nn.Linear(512, n_classes))
            ]))

        if self.device:
            self.to(self.device)

    def forward(self, input):
        return self.classifier(self.features(input))

    def weights_init(self, init_list, vae_list, flow_list=None, pretrained=None, filters_list=None, logvar=-10.):
        if len(init_list) == 1 and init_list[0] == 'no':
            return

        self.apply(utils.weight_init(module=nn.Conv2d, initf=nn.init.xavier_normal_))
        self.apply(utils.weight_init(module=nn.Linear, initf=nn.init.xavier_normal_))
        self.apply(utils.weight_init(module=bayes.LogScaleConv2d, initf=utils.const_init(logvar)))
        self.apply(utils.weight_init(module=bayes.LogScaleLinear, initf=utils.const_init(logvar)))

        if len(init_list) > 0 and init_list[0] == 'pretrained':
            assert len(init_list) == 1
            w_pretrained = torch.load(pretrained)
            for k, v in w_pretrained.items():
                if k in self.state_dict():
                    self.state_dict()[k].data.copy_(v)
                else:
                    tokens = k.split('.')
                    self.state_dict()['.'.join(tokens[:2] + ['mean'] + tokens[-1:])].data.copy_(v)
            return

        convs = [self.features.conv1, self.features.conv2, self.features.conv3]
        for i, m in enumerate(convs):
            init = init_list[i] if i < len(init_list) else 'xavier'
            w = m.mean.weight if isinstance(m, bayes._Bayes) else m.weight
            if init == 'vae':
                vae_path = vae_list[i]
                vae = utils.load_vae(vae_path, device=self.device)
                z = torch.randn(w.size(0) * w.size(1), vae.encoder.z_dim, 1, 1).to(vae.device)
                x = vae.decode(z)[0]
                w.data = x.reshape(w.shape)
            elif init == 'flow':
                flow_path = flow_list[i]
                flow = utils.load_flow(flow_path, device=self.device)
                utils.flow_init(flow)(w)
            elif init == 'xavier' or init == 'no':
                pass
            elif init == 'filters':
                filters = np.load(filters_list[i])
                filters = np.concatenate([filters]*10)
                N = np.prod(w.shape[:2])
                filters = filters[np.random.permutation(len(filters))[:N]]
                w.data = torch.from_numpy(filters.reshape(*w.shape)).to(self.device)
            elif init == 'recon':
                filters = np.load(filters_list[i])
                filters = np.concatenate([filters]*10)
                N = np.prod(w.shape[:2])
                filters = filters[np.random.permutation(len(filters))[:N]]
                vae_path = vae_list[i]
                vae = utils.load_vae(vae_path, device=self.device)
                filters = vae(torch.from_numpy(filters).to(self.device))[1][0]
                w.data = filters.reshape_as(w)
            else:
                raise NotImplementedError('no {} init'.format(init))

    def set_prior(self, prior_list, dwp_samples, vae_list, flow_list=None):
        convs = [self.features.conv1, self.features.conv2, self.features.conv3]
        for i, m in enumerate(convs):
            if not isinstance(m, bayes._Bayes):
                continue

            if prior_list[i] == 'vae':
                vae = utils.load_vae(vae_list[i], self.device)
                vae = nn.DataParallel(vae)
                self.vaes.append(vae)
                for p in vae.parameters():
                    p.requires_grad = False
                m.kl_function = utils.kl_dwp(vae, n_tries=dwp_samples)
            elif prior_list[i] == 'flow':
                flow = utils.load_flow(flow_list[i], self.device)
                for p in flow.parameters():
                    p.requires_grad = False
                m.kl_function = utils.kl_flow(flow, n_tries=dwp_samples)
            elif prior_list[i] == 'sn':
                m.kl_function = utils.kl_normal
                m.prior = dist.Normal(torch.FloatTensor([0.]).to(self.device),
                                      torch.FloatTensor([1.]).to(self.device))
            elif prior_list[i] == 'loguniform':
                if self.cfg in ['bayes111-mutrunca']:
                    m.kl_function = utils.kl_loguniform_with_trunc_alpha
                else:
                    raise NotImplementedError
            elif prior_list[i] == 'no':
                pass
            else:
                raise NotImplementedError

    def set_dwp_regularizer(self, vae_list):
        for path in vae_list:
            vae = utils.load_vae(path, device=self.device)
            for p in vae.parameters():
                p.requires_grad = False
            self.vaes.append(vae)

    def get_dwp_reg(self, backward=False, n_tries=1, weight=1., target='elbo'):
        modules = [self.features.conv1, self.features.conv2, self.features.conv3]
        reg = 0.
        for m, vae in zip(modules, self.vaes):
            reg += utils.dwp_regularizer(vae, m, n_tries=n_tries, backward=backward, weight=weight, target=target)
        return reg
