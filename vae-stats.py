import torch
import utils
from utils import tonp
from models import vae as vae_module
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
from torch.autograd import Variable
import seaborn as sns
from scipy.stats import norm
sns.set()
plt.switch_backend('agg')


def kl_q_vae_ub(n_samples, eps=1e-6):
    w = np.concatenate([np.random.randn(*data.shape) * np.sqrt(eps) + data for _ in range(n_samples)])
    w_inp = torch.FloatTensor(w).to(device)
    z_mean, z_var = vae.encode(w_inp)
    z = Variable(torch.randn(*z_mean.shape).cuda()) * torch.sqrt(z_var) + z_mean
    w_mu, w_var = map(tonp, vae.decode(z))
    z_mean, z_var = map(tonp, [z_mean, z_var])

    entq = 0.5 * (np.log(2 * np.pi * eps) + 1) * k2
    logp_theta = norm.logpdf(w, loc=w_mu, scale=np.sqrt(w_var)).sum(1)
    kl_rp = 0.5 * (z_var + z_mean ** 2 - np.log(z_var) - 1).sum(1)

    kl_vae_upper_bound = (kl_rp - entq - logp_theta).sum() / (n_samples * 1.)

    return kl_vae_upper_bound


def kl_q_sn(eps=1e-6, lamb=1.):
    return (data**2 / lamb + eps/lamb + np.log(lamb/eps) - 1.).sum() * 0.5


def kl_randomq_vae_ub(n_samples, s=1e-6):
    w = np.concatenate([np.random.randn(*data.shape) * np.sqrt(s) for _ in range(n_samples)])
    w_inp = torch.FloatTensor(w).to(device)
    z_mean, z_var = vae.encode(w_inp)
    z = Variable(torch.randn(*z_mean.shape).cuda()) * torch.sqrt(z_var) + z_mean
    w_mu, w_var = map(tonp, vae.decode(z))
    z_mean, z_var = map(tonp, [z_mean, z_var])

    entq = 0.5 * (np.log(2 * np.pi * s) + 1) * k2
    logp_theta = norm.logpdf(w, loc=w_mu, scale=np.sqrt(w_var)).sum(1)
    kl_rp = 0.5 * (z_var + z_mean ** 2 - np.log(z_var) - 1).sum(1)

    return (kl_rp - entq - logp_theta).sum() / (n_samples * 1.)


def kl_randq_sn(s=1e-6, lamb=1.):
    return (s/lamb + np.log(lamb/s) - 1.) * np.prod(data.shape) * 0.5


parser = argparse.ArgumentParser()
parser.add_argument('--model')
parser.add_argument('--data')
parser.add_argument('--log-dir', default='./vaelog/')
parser.add_argument('--gpu-id', default='0')
parser.add_argument('--z-dim', default=8, type=int)
parser.add_argument('--hidden-dim', default=16, type=int)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
args.cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

data = np.load(args.data)
D = data.shape[-1]

if D == 3:
    decoder = vae_module.Decoder3x3(args.z_dim, args.hidden_dim)
    encoder = vae_module.Encoder3x3(args.z_dim, args.hidden_dim)
elif D == 5:
    decoder = vae_module.Decoder5x5(args.z_dim, args.hidden_dim)
    encoder = vae_module.Encoder5x5(args.z_dim, args.hidden_dim)

vae = vae_module.VAE(encoder, decoder, device=device)
vae.load_state_dict(torch.load(args.model))

# Reconstructions
n, m = 10, 5
np.random.seed(42)
fig, axes = plt.subplots(figsize=(15, 12), nrows=n, ncols=m)
img_idxs = np.random.randint(0, len(data), size=(n * m))

for i, ax in enumerate(axes.flat):
    c = data[img_idxs[i]][np.newaxis]
    inp = torch.FloatTensor(c).to(device)
    _, [rec, _] = vae(inp)
    rec = tonp(rec)
    c = np.concatenate([c, rec], 3)[0, 0]
    sns.heatmap(c, ax=ax)
    ax.axis('off')

plt.savefig(os.path.join(args.log_dir, 'reconstruction'), dpi=200)
plt.clf()

# Samples
n, m = 10, 10
z = torch.randn(n * m, args.z_dim, 1, 1).to(device)
mu, var = vae.decode(z)

x = tonp(mu)[:, 0]

fig, axes = plt.subplots(figsize=(5, 5), nrows=n, ncols=m)
for i, ax in enumerate(axes.flat):
    im = ax.imshow(x[i], cmap='viridis')
    ax.axis('off')

plt.savefig(os.path.join(args.log_dir, 'samples'), dpi=200)
plt.clf()

# KL comparison
k2 = D**2

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
eps_grid = np.logspace(-9, -3, 20)
kl_standart_normal = []
kl_vae_upper_bound = []
for eps in eps_grid:
    kl_vae_upper_bound.append(kl_q_vae_ub(100, eps=eps))
    kl_standart_normal.append(kl_q_sn(eps=eps))

plt.semilogx(eps_grid, np.log(np.asarray(kl_standart_normal) + 1), label=r'$KL(N(W, \varepsilon)\, || \,N(0, 1))$')
plt.semilogx(eps_grid, np.log(np.asarray(kl_vae_upper_bound) + 1), label=r'UB $KL(N(W, \varepsilon)\, ||\, p_{VAE})$')
plt.xlabel(r'$\varepsilon$')
plt.ylabel('log(x + 1)')
plt.legend()

plt.subplot(1, 2, 2)
s_grid = np.logspace(-8, -4, 9)
kl_rq_vae = []
kl_rq_sn = []
kl_standart_normal = []
kl_vae_upper_bound = []
for s in s_grid:
    kl_rq_vae.append(kl_randomq_vae_ub(100, s=s))
    kl_rq_sn.append(kl_randq_sn(s=s))
    kl_vae_upper_bound.append(kl_q_vae_ub(100, eps=s))
    kl_standart_normal.append(kl_q_sn(eps=s))

plt.semilogx(s_grid, np.log(np.asarray(kl_rq_vae) + 1), label=r'UB KL(N(0, $s$) || $p_{VAE}$ )')
plt.semilogx(s_grid, np.log(np.asarray(kl_vae_upper_bound) + 1), label=r'UB $KL(N(W, s)\, ||\, p_{VAE})$')

plt.semilogx(s_grid, np.log(np.asarray(kl_rq_sn) + 1), label=r'KL(N(0, $s$) || N(0, 1) )')
plt.semilogx(s_grid, np.log(np.asarray(kl_standart_normal) + 1), label=r'$KL(N(W, s)\, || \,N(0, 1))$')

plt.xlabel(r'$s$')
plt.ylabel('log(x + 1)')
plt.legend()

plt.savefig(os.path.join(args.log_dir, 'kl_plots'), dpi=200)
plt.clf()
