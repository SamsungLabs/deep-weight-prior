from torch import distributions as dist
import torch
from torch import nn
import torch.nn.functional as F
import utils
import models.vae as vae_mod
import yaml


class _BayesNet(nn.Module):
    """
    """
    def __init__(self):
        super(_BayesNet, self).__init__()

    def kl(self, backward=False, weight=1.):
        kl = 0.
        for m in self.modules():
            if isinstance(m, _Bayes):
                kl += m.klf(backward=backward, weight=weight)

        return kl

    def _set_prior(self, pname, **kwrags):
        prior = dist.Normal(torch.FloatTensor([kwrags.get('mean', 0.)]).to(self.device),
                            torch.FloatTensor([kwrags.get('std', 1.)]).to(self.device))
        dwp_samples = kwrags.get('dwp_samples', 1)

        if pname == 'sn':
            for m in self.modules():
                if isinstance(m, _Bayes):
                    m.kl_function = utils.kl_normal
                    m.prior = prior
        elif pname == 'sn-mc':
            for m in self.modules():
                if isinstance(m, _Bayes):
                    m.kl_function = utils.kl_normal_mc
                    m.prior = prior
        elif pname == 'dwp':
            vae = utils.load_vae(kwrags['vae'], self.device)

            for p in vae.parameters():
                p.requires_grad = False

            klf = utils.kl_dwp(vae, n_tries=dwp_samples)

            for m in self.modules():
                if isinstance(m, BayesConv2d):
                    m.kl_function = klf
                elif isinstance(m, _Bayes):
                    m.kl_function = utils.kl_normal
                    m.prior = prior
        else:
            raise NotImplementedError


class _Bayes(object):
    """
    Technical class for indicating eigther module is bayes or not.
    """
    _mode = 'stoch'
    kl_function = None
    prior = None

    def klf(self, backward=False, weight=1.):
        if self.kl_function is None:
            return 0.
        return self.kl_function(self, backward=backward, weight=weight)

    def dist(self):
        raise NotImplementedError


class BayesConv2d(nn.Module, _Bayes):
    def __init__(self, *params, **kwrags):
        super(BayesConv2d, self).__init__()
        self.mean = nn.Conv2d(*params, **kwrags)
        # self.var = PosConv2d(*params, bias=False, **kwrags)
        self.var = LogScaleConv2d(*params, bias=False, **kwrags)
        self.eps = 1e-12

    def forward(self, input):
        if self._mode == 'det':
            return self.mean(input)

        m, v = self.mean(input), self.var(input**2)
        return dist.Normal(m, torch.sqrt(v + self.eps)).rsample()

    def dist(self):
        # '+ 0' convert Parameter to Tensor
        return dist.Normal(self.mean.weight + 0., torch.sqrt(self.var.get_weight() + self.eps))

    def q_params(self):
        return self.mean.weight, torch.sqrt(self.var.get_weight() + self.eps)


class MuTruncAlphaFFGConv2d(nn.Module, _Bayes):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=True):

        super(MuTruncAlphaFFGConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.mean = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.w_alpha = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size).fill_(-2.))
        self.eps = 1e-12
        self.train_alpha = True

    def forward(self, input):
        if self._mode == 'det':
            return self.mean(input)

        m = self.mean(input)
        alpha = torch.sigmoid(self.w_alpha)
        if not self.train_alpha:
            alpha = alpha.detach()

        v = F.conv2d(input**2, alpha * (self.mean.weight ** 2), bias=None, stride=self.stride, padding=self.padding)

        return dist.Normal(m, torch.sqrt(v + self.eps)).rsample()

    def dist(self):
        # '+ 0' convert Parameter to Tensor
        sigma2 = self.get_alpha() * (self.mean.weight ** 2)
        return dist.Normal(self.mean.weight + 0., torch.sqrt(sigma2 + self.eps))

    def q_params(self):
        sigma2 = self.get_alpha() * (self.mean.weight ** 2)
        return self.mean.weight, torch.sqrt(sigma2 + self.eps)

    def get_alpha(self):
        return torch.sigmoid(self.w_alpha)


class FFGLinear(nn.Module, _Bayes):
    def __init__(self, *params, **kwrags):
        super(FFGLinear, self).__init__()
        self.mean = nn.Linear(*params, **kwrags)
        self.var = LogScaleLinear(*params, bias=False, **kwrags)
        self.eps = 0.

    def forward(self, input):
        if self._mode == 'det':
            return self.mean(input)

        m, v = self.mean(input), self.var(input**2)
        return dist.Normal(m, torch.sqrt(v + self.eps)).rsample()

    def dist(self):
        return dist.Normal(self.mean.weight, torch.sqrt(self.var.get_weight() + self.eps))

    def q_params(self):
        return self.mean.weight, torch.sqrt(self.var.get_weight() + self.eps)


class PosConv2d(nn.Conv2d):
    def __init__(self, *params, **kwrags):
        super(PosConv2d, self).__init__(*params, **kwrags)

    def forward(self, input):
        return F.conv2d(input, F.softplus(self.weight), self.bias,
                        self.stride, self.padding, self.dilation, self.groups)

    def get_weight(self):
        return F.softplus(self.weight)


class LogScaleConv2d(nn.Conv2d):
    def __init__(self, *params, **kwrags):
        super(LogScaleConv2d, self).__init__(*params, **kwrags)

    def forward(self, input):
        return F.conv2d(input, torch.exp(self.weight), self.bias,
                        self.stride, self.padding, self.dilation, self.groups)

    def get_weight(self):
        return torch.exp(self.weight)


class LogScaleLinear(nn.Linear):
    def __init__(self, *params, **kwrags):
        super(LogScaleLinear, self).__init__(*params, **kwrags)

    def forward(self, input):
        return F.linear(input, torch.exp(self.weight), self.bias)

    def get_weight(self):
        return torch.exp(self.weight)


class FFGDWRConv2d(nn.Module, _Bayes):
    def __init__(self, in_channels, out_channels, kernel_size, vae, bias=True,
                 logvar=-10.):
        super(FFGDWRConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.decoder = nn.DataParallel(vae.decoder)
        for p in self.decoder.parameters():
            p.requires_grad = False

        self.z_dim = vae.encoder.z_dim
        self.z_mu = nn.Parameter(torch.randn(out_channels, in_channels, self.z_dim))
        self.z_logvar = nn.Parameter(torch.Tensor(out_channels, in_channels, self.z_dim).fill_(logvar))
        self.weight = None

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels).zero_())
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        return F.conv2d(input, self.get_weights(), self.bias)

    def get_weights(self):
        z = self.dist().rsample()
        w = self.decoder(z.view((-1, self.z_dim, 1, 1)))[0]
        w = w.view((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        return w

    def dist(self):
        return dist.Normal(self.z_mu + 0., torch.sqrt(torch.exp(self.z_logvar)))

    def q_params(self):
        return self.z_mu, torch.sqrt(torch.exp(self.z_logvar))
