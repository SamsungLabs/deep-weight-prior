import torch
from torch import nn
import utils
import numpy as np
import os
import time
from models.lenet import FConvMNIST
from models.cifarnet import CIFARNet, CIFARNetNew
import utils
from logger import Logger
from utils import tonp
import sys
from torch import distributions as dist
import models
from models.bayes import _Bayes, BayesConv2d, FFGLinear
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
parser.add_argument('--name', default='')
parser.add_argument('--data', default='cifar')
parser.add_argument('--num_examples', default=None, type=int)
parser.add_argument('--data_split_seed', default=42, type=int)
parser.add_argument('--resume', default='')
parser.add_argument('--lr', default=0.001, type=float, help='Initial learning rate')
parser.add_argument('--gamma', default=0.5, type=float)
parser.add_argument('--do', default=[], type=float, nargs='*')
parser.add_argument('--epochs', default=100, type=int, help='Number of epochs')
parser.add_argument('--decrease_from', default=0, type=int)
parser.add_argument('--bs', default=256, type=int, help='Batch size')
parser.add_argument('--test_bs', default=500, type=int, help='Batch size for test dataloader')
parser.add_argument('--model', default='vgg')
parser.add_argument('--model_size', default=1., type=float)
parser.add_argument('--pretrained', default='')
parser.add_argument('--filters_list', default=[], nargs='*', type=str)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--init', default='xavier')
parser.add_argument('--init_list', type=str, nargs='*', default=[])
parser.add_argument('--vae', default='')
parser.add_argument('--vae_list', type=str, nargs='*', default=[])
parser.add_argument('--milestones', type=int, nargs='*', default=[])
parser.add_argument('--net_cfg', default='E')
parser.add_argument('--hid_dim', default=[32, 64], type=int, nargs='+')
parser.add_argument('--n_classes', default=10, type=int)
parser.add_argument('--l2', default=0., type=float)
parser.add_argument('--eval_freq', default=1, type=int)
parser.add_argument('--dwp_reg', default=0., type=float)
parser.add_argument('--dwp_samples', default=1, type=int)
parser.add_argument('--rfe', default=0, type=int)
parser.add_argument('--fastconv', default=0, type=int)
parser.add_argument('--aug', default=0, type=int)

args = parser.parse_args()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.cuda.manual_seed_all(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

fmt = {
    'lr': '.5f',
    'sec': '.0f',
}
logger = Logger('logs', base=args.root, fmt=fmt)

# Load Datasets
trainloader, testloader = utils.load_dataset(data=args.data, train_bs=args.bs, test_bs=args.test_bs,
                                             num_examples=args.num_examples, seed=args.data_split_seed,
                                             augmentation=(args.aug == 1))

if args.model == 'fconv':
    net = FConvMNIST(args.net_cfg, device=device, hid_dim=args.hid_dim, do=args.do)
elif args.model == 'cifarnet':
    net = CIFARNet(args.net_cfg, device=device, n_classes=args.n_classes, do=args.do, k=args.model_size)
elif args.model == 'cifarnetnew':
    net = CIFARNetNew(args.net_cfg, device=device, n_classes=args.n_classes, do=args.do, k=args.model_size,
                      vae_list=args.vae_list)
else:
    raise NotImplementedError

# Initialization
if hasattr(net, 'weights_init'):
    net.weights_init(args.init_list, args.vae_list, pretrained=args.pretrained, filters_list=args.filters_list)
else:
    utils.net_init(net, args.init, args.vae)

if args.dwp_reg != 0:
    net.set_dwp_regularizer(args.vae_list)

# Optimizer
train_params = []
if args.rfe == 0:
    train_params = net.parameters()
elif args.rfe == 1:
    tr_modeules = [net.classifier]
    train_params = list(net.classifier.parameters())
    for m in net.features.modules():
        if isinstance(m, nn.BatchNorm2d):
            train_params += list(m.parameters())
            tr_modeules += [m]
    print('==> Random Feature Extration mode')
    print(*tr_modeules)
else:
    raise NotImplementedError

opt = torch.optim.Adam(train_params, lr=args.lr)
lrscheduler = MultiStepLR(opt, args.milestones, gamma=args.gamma)

# Load params if fine-tuning
if args.resume:
    net.load_state_dict(torch.load(os.path.join(args.resume, 'model.torch')))
    opt.load_state_dict(torch.load(os.path.join(args.resume, 'opt.torch')))

N = len(trainloader.dataset)
t0 = time.time()

it = 0
for e in range(1, args.epochs + 1):
    if args.milestones:
        lrscheduler.step()
    else:
        adjust_learning_rate(opt, lr_linear(e - 1))
    net.train()
    train_acc = utils.MovingMetric()
    train_nll = utils.MovingMetric()
    train_loss = utils.MovingMetric()
    opt.zero_grad()

    for x, y in trainloader:
        opt.zero_grad()
        it += 1
        x = x.to(device)
        y = y.to(device)

        p = net(x)

        data_term = F.cross_entropy(p, y)
        l2_norm = torch.FloatTensor([0.]).to(device)
        if args.l2 != 0:
            l2_norm = torch.sum(torch.stack([torch.sum(p**2) for p in net.features.parameters()]))

        dwp_reg = 0.
        if args.dwp_reg != 0.:
            dwp_reg = net.get_dwp_reg(backward=True, n_tries=args.dwp_samples, weight=args.dwp_reg)

        loss = data_term + args.l2 * l2_norm

        loss.backward()

        opt.step()

        loss += args.dwp_reg * dwp_reg

        acc = torch.sum(p.max(1)[1] == y)
        train_acc.add(acc.item(), p.size(0))
        train_nll.add(data_term.item() * x.size(0), x.size(0))
        train_loss.add(loss.item() * x.size(0), x.size(0))

        if args.fastconv == 1:
            if (it % args.eval_freq) == 0 or it == 1:
                net.eval()

                logp_test, labels = predict(testloader, net)
                test_acc = np.mean(logp_test.argmax(1) == labels)
                test_nll = -logp_test[np.arange(len(labels)), labels].mean()

                logger.add_scalar(it, 'loss', train_loss.get_val())
                logger.add_scalar(it, 'train_nll', train_nll.get_val())
                logger.add_scalar(it, 'test_nll', test_nll)
                logger.add_scalar(it, 'train_acc', train_acc.get_val())
                logger.add_scalar(it, 'test_acc', test_acc)
                logger.add_scalar(it, 'lr', opt.param_groups[0]['lr'])
                logger.add_scalar(it, 'sec', time.time() - t0)
                logger.add_scalar(it, 'l2_norm', l2_norm.item())

                logger.iter_info()
                logger.save()

                torch.save(net.state_dict(), os.path.join(args.root, 'model.torch'))
                torch.save(opt.state_dict(), os.path.join(args.root, 'opt.torch'))

                t0 = time.time()

                net.train()

    if ((e % args.eval_freq) == 0 or e == 1) and (args.fastconv == 0):
        net.eval()

        logp_test, labels = predict(testloader, net)
        test_acc = np.mean(logp_test.argmax(1) == labels)
        test_nll = -logp_test[np.arange(len(labels)), labels].mean()

        logger.add_scalar(e, 'loss', train_loss.get_val())
        logger.add_scalar(e, 'train_nll', train_nll.get_val())
        logger.add_scalar(e, 'test_nll', test_nll)
        logger.add_scalar(e, 'train_acc', train_acc.get_val())
        logger.add_scalar(e, 'test_acc', test_acc)
        logger.add_scalar(e, 'lr', opt.param_groups[0]['lr'])
        logger.add_scalar(e, 'sec', time.time() - t0)
        logger.add_scalar(e, 'l2_norm', l2_norm.item())
        logger.add_scalar(e, 'dwp_reg', dwp_reg)

        logger.iter_info()
        logger.save()

        torch.save(net.state_dict(), os.path.join(args.root, 'model.torch'))
        torch.save(opt.state_dict(), os.path.join(args.root, 'opt.torch'))

        t0 = time.time()

torch.save(net.state_dict(), os.path.join(args.root, 'model.torch'))
torch.save(opt.state_dict(), os.path.join(args.root, 'opt.torch'))
