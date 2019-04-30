import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.distributions as dist


def rnormal(mean, sigma):
    eps = Variable(torch.randn(mean.shape))
    if mean.data.is_cuda:
        eps = eps.cuda()
    return eps * sigma + mean


class VAE(nn.Module):
    def __init__(self, encoder, decoder, device=None, use_cuda=True):
        super(VAE, self).__init__()
        # depricated
        self.use_cuda = use_cuda

        self.encoder = encoder
        self.decoder = decoder

        self.device = device
        self.to(device)

    def encode(self, input):
        return self.encoder(input)

    def decode(self, input):
        return self.decoder(input)

    def forward(self, input):
        mu, var = self.encoder(input)
        z = dist.Normal(mu, torch.sqrt(var)).rsample()

        return (mu, var), self.decoder(z)


class DecoderFC(nn.Module):
    def __init__(self, z_dim, hidden_dim, out_dim):
        super(DecoderFC, self).__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )

        self.fc_mu = nn.Linear(hidden_dim, out_dim)
        self.fc_var = nn.Linear(hidden_dim, out_dim)

    def forward(self, input):
        x = self.decoder(input)
        mu = self.fc_mu(x)
        var = F.softplus(self.fc_var(x))
        return mu, var


class EncoderFC(nn.Module):
    def __init__(self, z_dim, hidden_dim, input_dim):
        super(EncoderFC, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_sigma = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        hidden = self.features(x)

        z_mu = self.fc_mu(hidden)
        z_var = F.softplus(self.fc_sigma(hidden))
        return z_mu, z_var


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view((x.size(0), -1))


class UnFlatten(nn.Module):
    def __init__(self, w=1):
        super(UnFlatten, self).__init__()
        self.w = w

    def forward(self, x):
        return x.view((x.size(0), -1, self.w, self.w))


class Decoder7x7(nn.Module):
    def __init__(self, z_dim, hidden_dim, var='train'):
        super(Decoder7x7, self).__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(z_dim, hidden_dim * 2, 3),
            nn.ELU(),
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim * 2, 3),
            nn.ELU(),
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 3),
            nn.ELU(),
        )

        self.fc_mu = nn.Conv2d(hidden_dim, 1, 1)
        self.fc_var = nn.Conv2d(hidden_dim, 1, 1)

    def forward(self, input):
        x = self.decoder(input)
        mu = self.fc_mu(x)
        var = F.softplus(self.fc_var(x))
        return mu, var


class Encoder7x7(nn.Module):
    def __init__(self, z_dim, hidden_dim, var='train'):
        super(Encoder7x7, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, hidden_dim, 3),
            nn.ELU(),
            nn.Conv2d(hidden_dim, hidden_dim * 2, 3),
            nn.ELU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, 3),
            nn.ELU(),
        )
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.fc_mu = nn.Conv2d(hidden_dim * 2, z_dim, 1)
        self.fc_var = nn.Conv2d(hidden_dim * 2, z_dim, 1)

    def forward(self, x):
        hidden = self.features(x)
        z_mu = self.fc_mu(hidden)
        z_var = F.softplus(self.fc_var(hidden))
        return z_mu, z_var


class Decoder3x3(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super(Decoder3x3, self).__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim

        self.decoder = nn.Sequential(
            nn.Conv2d(z_dim, hidden_dim * 2, 1),
            nn.ELU(),
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 3),
            nn.ELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.ELU(),
        )

        self.fc_mu = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, 1),
        )
        self.fc_var = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, 1)
        )

    def forward(self, input):
        x = self.decoder(input)
        mu = self.fc_mu(x)
        var = Variable(torch.FloatTensor([1e-3]).cuda())  # F.softplus(self.fc_var(x))
        return mu, var


class Encoder3x3(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super(Encoder3x3, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, hidden_dim, 3, padding=1),
            nn.ELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ELU(),
            nn.Conv2d(hidden_dim, hidden_dim*2, 3),
            nn.ELU(),
        )
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.fc_mu = nn.Conv2d(hidden_dim * 2, z_dim, 1)
        self.fc_sigma = nn.Conv2d(hidden_dim * 2, z_dim, 1)

    def forward(self, x):
        hidden = self.features(x)

        z_mu = self.fc_mu(hidden)
        z_var = F.softplus(self.fc_sigma(hidden))
        return z_mu, z_var


class Decoder5x5(nn.Module):
    def __init__(self, z_dim, hidden_dim, var='train'):
        super(Decoder5x5, self).__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.activation = nn.ELU()

        if var != 'train':
            var = float(var)
        self.var = var

        self.decoder = nn.Sequential(
            nn.Conv2d(z_dim, hidden_dim * 2, 1),
            self.activation,
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim * 2, 3),
            self.activation,
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim * 2, 3),
            self.activation,
            nn.Conv2d(hidden_dim * 2, hidden_dim, 1),
            self.activation,
        )

        self.fc_mu = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, 1),
        )
        self.fc_var = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, 1)
        )

    def forward(self, input):
        x = self.decoder(input)
        mu = self.fc_mu(x)
        if isinstance(self.var, float):
            var = torch.FloatTensor([self.var]).cuda()
        else:
            var = F.softplus(self.fc_var(x))

        return mu, var


class Encoder5x5(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super(Encoder5x5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, hidden_dim, 3, padding=1),
            nn.ELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ELU(),
            nn.Conv2d(hidden_dim, hidden_dim*2, 3),
            nn.ELU(),
            nn.Conv2d(hidden_dim*2, hidden_dim*2, 3),
            nn.ELU(),
        )
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.fc_mu = nn.Conv2d(hidden_dim * 2, z_dim, 1)
        self.fc_sigma = nn.Conv2d(hidden_dim * 2, z_dim, 1)

    def forward(self, x):
        hidden = self.features(x)

        z_mu = self.fc_mu(hidden)
        z_var = F.softplus(self.fc_sigma(hidden))
        return z_mu, z_var


class Encoder16x16(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super(Encoder16x16, self).__init__()
        self.features = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, hidden_dim, 3),
            nn.MaxPool2d(2),
            nn.ELU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, hidden_dim * 2, 3),
            nn.MaxPool2d(2),
            nn.ELU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, 3),
            nn.MaxPool2d(2),
            nn.ELU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, 3),
            nn.MaxPool2d(2),
            nn.ELU(),
        )
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.fc_mu = nn.Conv2d(hidden_dim * 2, z_dim, 1)
        self.fc_sigma = nn.Conv2d(hidden_dim * 2, z_dim, 1)

    def forward(self, x):
        hidden = self.features(x)

        z_mu = self.fc_mu(hidden)
        z_var = F.softplus(self.fc_sigma(hidden))
        return z_mu, z_var


class Decoder16x16(nn.Module):
    def __init__(self, z_dim, hidden_dim, var='train'):
        super(Decoder16x16, self).__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.activation = nn.ELU()

        if var != 'train':
            var = float(var)
        self.var = var

        self.decoder = nn.Sequential(
            nn.Conv2d(z_dim, hidden_dim * 2, 1),
            self.activation,
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim * 2, 4, stride=2),
            self.activation,
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim * 2, 4, stride=2, padding=1),
            self.activation,
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 4, stride=2, padding=1),
            self.activation,
        )

        self.fc_mu = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, 1),
        )
        self.fc_var = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, 1)
        )

    def forward(self, input):
        x = self.decoder(input)
        mu = self.fc_mu(x)
        if isinstance(self.var, float):
            var = torch.FloatTensor([self.var]).cuda()
        else:
            var = F.softplus(self.fc_var(x))

        return mu, var


class DecoderId(nn.Module):
    def __init__(self, out_dim):
        super(DecoderId, self).__init__()
        self.out_dim = out_dim

    def forward(self, input):
        d = input.device
        return torch.zeros(input.size(0), *self.out_dim).to(d), torch.ones(input.size(0), *self.out_dim).to(d)


class EncoderId(nn.Module):
    def __init__(self, z_dim):
        super(EncoderId, self).__init__()
        self.z_dim = z_dim

    def forward(self, x):
        d = x.device
        return torch.zeros(x.size(0), self.z_dim, 1, 1).to(d), torch.ones(x.size(0), self.z_dim, 1, 1).to(d)
