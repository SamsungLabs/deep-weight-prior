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


class FConvMNIST(bayes._BayesNet):
    def __init__(self, cfg, device=None, do=None, hid_dim=None, **kwargs):
        self.device = device
        super(FConvMNIST, self).__init__()
        self.hid_dim = hid_dim
        self.cfg = cfg

        if cfg == 'bayes':
            self.features = nn.Sequential(OrderedDict([
                ('conv1', bayes.BayesConv2d(1, hid_dim[0], 7)),
                ('relu1', nn.LeakyReLU()),
                ('mp1', nn.MaxPool2d(2)),

                ('conv2', bayes.BayesConv2d(hid_dim[0], hid_dim[1], 5)),
                ('relu2', nn.LeakyReLU()),
                ('mp2', nn.MaxPool2d(2)),

                ('flatten', Flatten())
            ]))
        elif cfg == 'bayes-mtrunca':
            self.features = nn.Sequential(OrderedDict([
                ('conv1', bayes.MuTruncAlphaFFGConv2d(1, hid_dim[0], 7)),
                ('relu1', nn.LeakyReLU()),
                ('mp1', nn.MaxPool2d(2)),

                ('conv2', bayes.MuTruncAlphaFFGConv2d(hid_dim[0], hid_dim[1], 5)),
                ('relu2', nn.LeakyReLU()),
                ('mp2', nn.MaxPool2d(2)),

                ('flatten', Flatten())
            ]))
        elif cfg == 'bayes-1-0':
            self.features = nn.Sequential(OrderedDict([
                ('conv1', bayes.BayesConv2d(1, hid_dim[0], 7)),
                ('relu1', nn.LeakyReLU()),
                ('mp1', nn.MaxPool2d(2)),

                ('conv2', nn.Conv2d(hid_dim[0], hid_dim[1], 5)),
                ('relu2', nn.LeakyReLU()),
                ('mp2', nn.MaxPool2d(2)),

                ('flatten', Flatten())
            ]))
        elif cfg == 'vanilla':
            self.features = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(1, hid_dim[0], 7)),
                ('relu1', nn.LeakyReLU()),
                ('mp1', nn.MaxPool2d(2)),

                ('conv2', nn.Conv2d(hid_dim[0], hid_dim[1], 5)),
                ('relu2', nn.LeakyReLU()),
                ('mp2', nn.MaxPool2d(2)),

                ('flatten', Flatten())
            ]))
        elif cfg == 'vanilla-do':

            self.features = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(1, hid_dim[0], 7)),
                ('relu1', nn.LeakyReLU()),
                ('do1', nn.Dropout(do[0])),
                ('mp1', nn.MaxPool2d(2)),

                ('conv2', nn.Conv2d(hid_dim[0], hid_dim[1], 5)),
                ('relu2', nn.LeakyReLU()),
                ('do2', nn.Dropout(do[1])),
                ('mp2', nn.MaxPool2d(2)),

                ('flatten', Flatten())
            ]))
        else:
            raise NotImplementedError

        self.classifier = nn.Linear(hid_dim[1] * 9, 10)

        if device:
            self.to(device)

    def weights_init(self, init_list, vae_list, flow_list=None, pretrained=None, filters_list=None, logvar=-10.):
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

        convs = [self.features.conv1, self.features.conv2]
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
                N = np.prod(w.shape[:2])
                filters = filters[np.random.permutation(len(filters))[:N]]
                w.data = torch.from_numpy(filters.reshape(*w.shape)).to(self.device)
            else:
                raise NotImplementedError

    def set_prior(self, prior_list, dwp_samples, vae_list, flow_list=None):
        convs = [self.features.conv1, self.features.conv2]
        for i, m in enumerate(convs):
            if not isinstance(m, bayes._Bayes):
                continue

            if prior_list[i] == 'vae':
                vae = utils.load_vae(vae_list[i], self.device)
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
                if self.cfg == 'bayes-mtrunca':
                    m.kl_function = utils.kl_loguniform_with_trunc_alpha
            else:
                raise NotImplementedError

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
