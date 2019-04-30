import os
import argparse
import numpy as np
import torch
from torch.autograd import Variable
import utils
from torch import nn
from utils import tonp
from logger import Logger
from models import vae
import sys
import torch.distributions as dist
# import matplotlib.pyplot as plt
# import seaborn as sns
import myexman
from pathlib import Path

# sns.set()
# plt.switch_backend('agg')


def train(trainloader, testloader, vae, optimizer, scheduler, criterion, args, D):
    logger = Logger(name='logs', base=args.root)
    prior = dist.Normal(torch.FloatTensor([0.]).to(vae.device), torch.FloatTensor([1.]).to(vae.device))
    for epoch in range(1, args.num_epochs + 1):
        adjust_learning_rate(optimizer, lr_linear(epoch))
        scheduler.step()
        train_likelihood = utils.MovingMetric()
        train_kl = utils.MovingMetric()

        for i, x in enumerate(trainloader):
            optimizer.zero_grad()
            x = x.to(vae.device)
            [z_mu, z_var], [x_mu, x_var] = vae(x)
            likelihood = dist.Normal(x_mu, torch.sqrt(x_var)).log_prob(x).sum()
            kl = dist.kl_divergence(dist.Normal(z_mu, torch.sqrt(z_var)), prior).sum()
            loss = -likelihood + kl

            loss.backward()
            optimizer.step()

            train_likelihood.add(likelihood.item(), x.size(0))
            train_kl.add(kl.item(), x.size(0))

        test_likelihood = utils.MovingMetric()
        test_kl = utils.MovingMetric()
        for i, x in enumerate(testloader):
            x = x.to(vae.device)
            [z_mu, z_var], [x_mu, x_var] = vae(x)

            test_likelihood.add(dist.Normal(x_mu, torch.sqrt(x_var)).log_prob(x).sum().item(), x.size(0))
            test_kl.add(dist.kl_divergence(dist.Normal(z_mu, torch.sqrt(z_var)), prior).sum().item(), x.size(0))

        test_likelihood = test_likelihood.get_val()
        test_kl = test_kl.get_val()
        train_likelihood = train_likelihood.get_val()
        train_kl = train_kl.get_val()

        logger.add_scalar(epoch, 'train_elbo', train_likelihood - train_kl)
        logger.add_scalar(epoch, 'train_ll', train_likelihood)
        logger.add_scalar(epoch, 'train_kl', train_kl)

        logger.add_scalar(epoch, 'test_elbo', test_likelihood - test_kl)
        logger.add_scalar(epoch, 'test_ll', test_likelihood)
        logger.add_scalar(epoch, 'test_kl', test_kl)

        logger.iter_info()
        logger.save()

        # if epoch % args.eval_freq == 0:
        #     z = prior.rsample(sample_shape=(25, args.z_dim, 1))
        #     x_mu, x_var = vae.decode(z)

        #     f, axs = plt.subplots(nrows=5, ncols=5, figsize=(12, 10))
        #     samples = dist.Normal(x_mu, torch.sqrt(x_var)).rsample()
        #     for x, ax in zip(samples.reshape((-1, D, D)), axs.flat):
        #         sns.heatmap(tonp(x), ax=ax)
        #         ax.axis('off')
        #     f.savefig(os.path.join(args.root, 'samples'), dpi=200)
        #     plt.close(f)

        #     data = next(iter(testloader))
        #     data = data[:25].to(vae.device)
        #     [z_mu, z_var], [x_mu, x_var] = vae(data)
        #     f, axs = plt.subplots(nrows=5, ncols=5, figsize=(15, 7))
        #     for x, x_rec, ax in zip(data.reshape((-1, D, D)), x_mu.reshape((-1, D, D)), axs.flat):
        #         sns.heatmap(np.concatenate((tonp(x), tonp(x_rec)), 1), ax=ax)
        #         ax.axis('off')
        #     f.savefig(os.path.join(args.root, 'mean_reconstructions'), dpi=200)
        #     plt.close(f)

        #     [z_mu, z_var], [x_mu, x_var] = vae(data)
        #     f, axs = plt.subplots(nrows=5, ncols=5, figsize=(15, 7))
        #     samples = dist.Normal(x_mu, torch.sqrt(x_var)).rsample()
        #     for x, x_rec, ax in zip(data.reshape((-1, D, D)), samples.reshape((-1, D, D)), axs.flat):
        #         sns.heatmap(np.concatenate((tonp(x), tonp(x_rec)), 1), ax=ax)
        #         ax.axis('off')
        #     f.savefig(os.path.join(args.root, 'sample_reconstructions'), dpi=200)
        #     plt.close(f)

        torch.save(vae.state_dict(), os.path.join(args.root, 'vae_params.torch'))
        torch.save(optimizer.state_dict(), os.path.join(args.root, 'opt_params.torch'))


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def lr_linear(epoch):
    lr = args.lr * np.minimum((-epoch) * 1. / (args.num_epochs) + 1, 1.)
    return max(0, lr)


if __name__ == '__main__':
    parser = myexman.ExParser(file=__file__)
    parser.add_argument('--train')
    parser.add_argument('--test')
    parser.add_argument('--data_dir', default='')
    parser.add_argument('--resume_vae', default='')
    parser.add_argument('--resume_opt', default='')
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--eval_freq', default=1, type=int)
    parser.add_argument('--num_epochs', default=1000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--test_bs', default=512, type=int)
    parser.add_argument('--z_dim', default=8, type=int)
    parser.add_argument('--hidden_dim', default=16, type=int)
    parser.add_argument('--kernel_dim', default=16, type=int)
    parser.add_argument('--verbose', default=1, type=int)
    parser.add_argument('--gpu_id', default='0')
    parser.add_argument('--lr_decay_step', default=int(11e8), type=int)
    parser.add_argument('--decay', default=0.5, type=float)
    parser.add_argument('--var', default='train')
    parser.add_argument('--name', default='')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    args.cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.data_dir:
        trainloader, D = utils.get_dataloader(os.path.join(args.data_dir, 'train.npy'), args.batch_size, shuffle=True)
        testloader, D = utils.get_dataloader(os.path.join(args.data_dir, 'test.npy'), args.test_bs, shuffle=False)
    else:
        trainloader, D = utils.get_dataloader(args.train, args.batch_size, shuffle=True)
        testloader, D = utils.get_dataloader(args.test, args.test_bs, shuffle=False)

    assert args.kernel_dim == D, '--kernel-dim != D (in dataset)'

    if D == 3:
        decoder = vae.Decoder3x3(args.z_dim, args.hidden_dim)
        encoder = vae.Encoder3x3(args.z_dim, args.hidden_dim)
    elif D == 5:
        decoder = vae.Decoder5x5(args.z_dim, args.hidden_dim, var=args.var)
        encoder = vae.Encoder5x5(args.z_dim, args.hidden_dim)
    elif D == 7:
        decoder = vae.Decoder7x7(args.z_dim, args.hidden_dim, var=args.var)
        encoder = vae.Encoder7x7(args.z_dim, args.hidden_dim)
    elif D == 16:
        decoder = vae.Decoder16x16(args.z_dim, args.hidden_dim, var=args.var)
        encoder = vae.Encoder16x16(args.z_dim, args.hidden_dim)

    vae = vae.VAE(encoder, decoder, device=device)
    if args.resume_vae:
        vae.load_state_dict(torch.load(args.resume_vae))
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr)
    if args.resume_opt:
        optimizer.load_state_dict(torch.load(args.resume_opt))
        optimizer.param_groups[0]['lr'] = args.lr
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.decay)
    criterion = utils.VAEELBOLoss(use_cuda=args.cuda)

    train(trainloader, testloader, vae, optimizer, scheduler, criterion, args, D)
