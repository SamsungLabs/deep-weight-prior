import torch
import utils
import numpy as np
import os
import time
from models.lenet import FConvMNIST
from models.cifarnet import CIFARNet, CIFARNetNew
import utils
from logger import Logger
import sys
import models.vae as vae_mod
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import myexman
from torch.nn import functional as F


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def lr_linear(epoch):
    lr = args.lr * np.minimum((args.decrease_from - epoch) * 1. / (args.epochs - args.decrease_from) + 1, 1.)
    return max(0, lr)


def predict(data, net):
    pred = []
    l = []
    for x, y in data:
        l.append(y.numpy())
        x = x.to(device)
        p = F.log_softmax(net(x), dim=1)
        pred.append(p.data.cpu().numpy())
    return np.concatenate(pred), np.concatenate(l)


parser = myexman.ExParser(file=__file__)
parser.add_argument('--data', default='cifar')
parser.add_argument('--num_examples', default=None, type=int)
parser.add_argument('--data_split_seed', default=42, type=int)
parser.add_argument('--resume', default='')
parser.add_argument('--lr', default=0.001, type=float, help='Initial learning rate')
parser.add_argument('--gamma', default=0.5, type=float)
parser.add_argument('--epochs', default=100, type=int, help='Number of epochs')
parser.add_argument('--decrease_from', default=0, type=int)
parser.add_argument('--bs', default=256, type=int, help='Batch size')
parser.add_argument('--test_bs', default=500, type=int, help='Batch size for test dataloader')
parser.add_argument('--model', default='vgg')
parser.add_argument('--model_size', default=1., type=float)
parser.add_argument('--vae', default='')
parser.add_argument('--vae_list', type=str, nargs='*', default=[])
parser.add_argument('--flow_list', type=str, nargs='*', default=[])
parser.add_argument('--vae_var', default='train')
parser.add_argument('--z_dim', default=8, type=int)
parser.add_argument('--dwp_samples', default=1, type=int)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--init', default='xavier')
parser.add_argument('--pretrained', default='')
parser.add_argument('--init_list', type=str, nargs='*', default=[])
parser.add_argument('--filters_list', default=[], nargs='*', type=str)
parser.add_argument('--milestones', type=int, nargs='*', default=[])
parser.add_argument('--net_cfg', default='bayes')
parser.add_argument('--hid_dim', default=[32, 64], type=int, nargs='+')
parser.add_argument('--prior', default='sn')
parser.add_argument('--prior_list', type=str, nargs='*', default=[])
parser.add_argument('--lamb', default=1., type=float)
parser.add_argument('--start_stoch', default=1, type=int)
parser.add_argument('--ens', default=10, type=int)
parser.add_argument('--n_classes', default=10, type=int)
parser.add_argument('--eval_freq', default=1, type=int)
parser.add_argument('--aug', default=0, type=int)
parser.add_argument('--logvarinit', default=-10, type=float)
parser.add_argument('--name', default='')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.cuda.manual_seed_all(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

fmt = {
    'lr': '.5f',
    'sec': '.0f',
    'KL': '.3g',
    'train_loss': '.3g',
    'test_loss': '.3g',
    'ens[{}]_test_acc'.format(args.ens): '.4f'
}

logger = Logger('logs', base=args.root, fmt=fmt)
param_logger = Logger('param-logs', base=args.root)

# Load Datasets
trainloader, testloader = utils.load_dataset(data=args.data, train_bs=args.bs, test_bs=args.test_bs,
                                             num_examples=args.num_examples, seed=args.data_split_seed,
                                             augmentation=(args.aug == 1))

if args.model == 'fconv':
    net = FConvMNIST(args.net_cfg, device=device, hid_dim=args.hid_dim)
elif args.model == 'cifarnet':
    net = CIFARNet(args.net_cfg, device=device, n_classes=args.n_classes, k=args.model_size)
elif args.model == 'cifarnetnew':
    net = CIFARNetNew(args.net_cfg, device=device, n_classes=args.n_classes, k=args.model_size,
                      vae_list=args.vae_list, logvar=args.logvarinit)
else:
    raise NotImplementedError

# Initialization
if hasattr(net, 'weights_init'):
    net.weights_init(args.init_list, args.vae_list, flow_list=args.flow_list, pretrained=args.pretrained,
                     filters_list=args.filters_list, logvar=args.logvarinit)
else:
    utils.net_init(net, args.init, args.vae)

# Define prior over _Bayes modules
if hasattr(net, "set_prior"):
    net.set_prior(args.prior_list, args.dwp_samples, args.vae_list, flow_list=args.flow_list)
else:
    net._set_prior(args.prior, dwp_samples=args.dwp_samples, std=args.lamb,
                   vae=args.vae)

# Optimizer
opt = torch.optim.Adam(net.parameters(), lr=args.lr)
lrscheduler = MultiStepLR(opt, args.milestones, gamma=args.gamma)

# Load params if fine-tuning
if args.resume:
    net.load_state_dict(torch.load(os.path.join(args.resume, 'model.torch')))
    opt.load_state_dict(torch.load(os.path.join(args.resume, 'opt.torch')))

N = len(trainloader.dataset)
t0 = time.time()


for e in range(1, args.epochs + 1):
    opt.zero_grad()

    # Learning rate stuff
    adjust_learning_rate(opt, lr_linear(e - 1))

    net.train()
    # det -- use mean, stoch -- sample weights
    utils.bnn_mode(net, 'stoch')

    mean_kl = utils.MovingMetric()
    train_acc = utils.MovingMetric()
    train_nll = utils.MovingMetric()
    kl_mean = utils.MovingMetric()

    for x, y in trainloader:
        opt.zero_grad()
        bs = x.size(0)
        x = x.to(device)
        y = y.to(device)
        p = net(x)

        data_term = F.cross_entropy(p, y, size_average=False)
        kl_term = 0

        # calculate (and backprop!) KL term if it is time to do
        kl_term = net.kl(backward=True)

        # normalize data term to be unbiased estimation of the SUM of LLs.
        (data_term * N / x.size(0)).backward()

        opt.step()

        acc = torch.sum(p.max(1)[1] == y)
        mean_kl.add(kl_term * x.size(0), x.size(0))
        train_acc.add(acc.item(), p.size(0))
        train_nll.add(data_term.item(), x.size(0))

    if (e % args.eval_freq) == 0 or e == 1:
        net.eval()
        # accuracy of the ensemble
        ens_pred, gt = utils.mc_ensemble(net, testloader, n_tries=args.ens, log=True)
        ensemble_acc = np.mean(ens_pred.argmax(1) == gt)
        test_nll = -ens_pred[np.arange(len(gt)), gt].mean()

        # one sample metrics
        samp_pred, gt = utils.mc_ensemble(net, testloader, n_tries=1, log=True)
        samp_acc = np.mean(samp_pred.argmax(1) == gt)

        utils.bnn_mode(net, 'det')

        # accuracy of deterministic net
        logp_test, labels = predict(testloader, net)
        test_acc = np.mean(logp_test.argmax(1) == labels)
        kl_term = utils.kl_term(net, backward=False)

        logger.add_scalar(e, 'train_nll', train_nll.get_val())
        logger.add_scalar(e, 'KL', mean_kl.get_val())
        logger.add_scalar(e, 'train_loss', train_nll.get_val() * N + mean_kl.get_val())
        logger.add_scalar(e, 'test_nll', test_nll)
        logger.add_scalar(e, 'test_loss', test_nll + kl_term)
        logger.add_scalar(e, 'train_acc', train_acc.get_val())
        logger.add_scalar(e, 'det_test_acc', test_acc)
        logger.add_scalar(e, 'samp_test_acc', samp_acc)
        logger.add_scalar(e, 'ens[{}]_test_acc'.format(args.ens), ensemble_acc)
        logger.add_scalar(e, 'lr', opt.param_groups[0]['lr'])
        logger.add_scalar(e, 'sec', time.time() - t0)

        logger.iter_info()
        logger.save()

        # variational approximation stats
        t0 = time.time()

        torch.save(net.state_dict(), os.path.join(args.root, 'model.torch'))
        torch.save(opt.state_dict(), os.path.join(args.root, 'opt.torch'))
