"""
    PyTorch training code for TFeat shallow convolutional patch descriptor:
    http://www.iis.ee.ic.ac.uk/%7Evbalnt/shallow_descr/TFeat_paper.pdf
    The code reproduces *exactly* it's lua anf TF version:
    https://github.com/vbalnt/tfeat
    2017 Edgar Riba
"""

from __future__ import print_function
import argparse
import torch
import torch.nn.init
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

import os
import cv2
import math
import numpy as np
from tqdm import tqdm

from eval_metrics import ErrorRateAt95Recall
from t_net import TNet
from logger import Logger
from data_loader import TripletPhotoTour

TFEAT_PATCH_SIZE = 32

# Training settings
parser = argparse.ArgumentParser(description='PyTorch TFeat Example')
# Model options
parser.add_argument('--dataroot', type=str, default='/tmp/phototour_dataset',
                    help='path to dataset')
parser.add_argument('--log-dir', default='./logs',
                    help='folder to output model checkpoints')
parser.add_argument('--imageSize', type=int, default=32,
                    help='the height / width of the input image to network')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=10, metavar='E',
                    help='number of epochs to train (default: 10)')
# Training options
parser.add_argument('--batch-size', type=int, default=128, metavar='BS',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='BST',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--n-triplets', type=int, default=1280000, metavar='N',
                    help='how many triplets will generate from the dataset')
parser.add_argument('--margin', type=float, default=2.0, metavar='MARGIN',
                    help='the margin value for the triplet loss function (default: 2.0')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--lr-decay', default=1e-6, type=float, metavar='LRD',
                    help='learning rate decay ratio (default: 1e-6')
parser.add_argument('--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--optimizer', default='sgd', type=str,
                    metavar='OPT', help='The optimizer to use (default: SGD)')
parser.add_argument('--anchorswap', type=bool, default=False,
                    help='turns on anchor swap')
# Device options
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=10, metavar='LI',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()

# set the device to use by setting CUDA_VISIBLE_DEVICES env variable in
# order to prevent any memory allocation on unused GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
cv2.setRNGSeed(args.seed)

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

if args.cuda:
    cudnn.benchmark = True
    torch.cuda.manual_seed_all(args.seed)

LOG_DIR = args.log_dir + '/run-optim_{}-n{}-lr{}-wd{}-m{}-S{}-tanh'\
    .format(args.optimizer, args.n_triplets, args.lr, args.wd,
            args.margin, args.seed)
# create logger
logger = Logger(LOG_DIR)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform(m.weight.data, gain=math.sqrt(2.0))
        nn.init.constant(m.bias.data, 0.1)

cv2_scale = lambda x: cv2.resize(x, dsize=(TFEAT_PATCH_SIZE, TFEAT_PATCH_SIZE),
                                     interpolation=cv2.INTER_LINEAR)
np_reshape = lambda x: np.reshape(x, (TFEAT_PATCH_SIZE, TFEAT_PATCH_SIZE, 1))

kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    TripletPhotoTour(train=True, root=args.dataroot, name='notredame',
                     download=True, n_triplets = args.n_triplets,transform=transforms.Compose([
            transforms.Lambda(cv2_scale),
            transforms.Lambda(np_reshape),
            transforms.ToTensor(),
            transforms.Normalize((0.48544601108437,), (0.18649942105166,))
        ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    TripletPhotoTour(train=False, root=args.dataroot, name='liberty',
                     download=True, n_triplets = args.n_triplets, transform=transforms.Compose([
            transforms.Lambda(cv2_scale),
            transforms.Lambda(np_reshape),
            transforms.ToTensor(),
            transforms.Normalize((0.48544601108437,), (0.18649942105166,))
        ])),
    batch_size=args.test_batch_size, shuffle=False, **kwargs)

# print the experiment configuration
print('\nparsed options:\n{}\n'.format(vars(args)))


def main():

    # instantiate model and initialize weights
    model = TNet()
    model.apply(weights_init)

    if args.cuda:
        model.cuda()

    optimizer = create_optimizer(model, args.lr)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print('=> no checkpoint found at {}'.format(args.resume))

    start = args.start_epoch
    end = start + args.epochs
    for epoch in range(start, end):
        train(train_loader, model, optimizer, epoch)
        test(test_loader, model, epoch)


def train(train_loader, model, optimizer, epoch):
    # switch to train mode
    model.train()

    pbar = tqdm(enumerate(train_loader))
    for batch_idx, (data_a, data_p, data_n) in pbar:

        if args.cuda:
            data_a, data_p, data_n = data_a.cuda(), data_p.cuda(), data_n.cuda()
        data_a, data_p, data_n = Variable(data_a), Variable(data_p), \
                                 Variable(data_n)

        # compute output
        out_a, out_p, out_n = model(data_a), model(data_p), model(data_n)
        loss = F.triplet_margin_loss(out_p, out_a, out_n, margin=args.margin, swap=args.anchorswap) 
        # compute gradient and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update the optimizer learning rate
        adjust_learning_rate(optimizer)

        # log loss value
        logger.log_value('loss', loss.data[0]).step()

        if batch_idx % args.log_interval == 0:
            pbar.set_description(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data_a), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.data[0]))

    # do checkpointing
    torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()},
               '{}/checkpoint_{}.pth'.format(LOG_DIR, epoch))


def test(test_loader, model, epoch):
    # switch to evaluate mode
    model.eval()

    labels, distances = [], []

    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (data_a, data_p, label) in pbar:
        if args.cuda:
            data_a, data_p = data_a.cuda(), data_p.cuda()
        data_a, data_p, label = Variable(data_a, volatile=True), \
                                Variable(data_p, volatile=True), Variable(label)

        # compute output
        out_a, out_p = model(data_a), model(data_p)
        dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
        distances.append(dists.data.cpu().numpy())
        labels.append(label.data.cpu().numpy())

        if batch_idx % args.log_interval == 0:
            pbar.set_description('Test Epoch: {} [{}/{} ({:.0f}%)]'.format(
                epoch, batch_idx * len(data_a), len(test_loader.dataset),
                100. * batch_idx / len(test_loader)))

    # measure accuracy (FPR95)
    num_tests = test_loader.dataset.matches.size(0)
    labels = np.vstack(labels).reshape(num_tests)
    distances = np.vstack(distances).reshape(num_tests)

    fpr95 = ErrorRateAt95Recall(labels, distances)
    print('\33[91mTest set: Accuracy(FPR95): {:.8f}\n\33[0m'.format(fpr95))

    logger.log_value('fpr95', fpr95)


def adjust_learning_rate(optimizer):
    """Updates the learning rate given the learning rate decay.
    The routine has been implemented according to the original Lua SGD optimizer
    """
    for group in optimizer.param_groups:
        if 'step' not in group:
            group['step'] = 0
        group['step'] += 1

        group['lr'] = args.lr / (1 + group['step'] * args.lr_decay)


def create_optimizer(model, new_lr):
    # setup optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=new_lr,
                              momentum=0.9, dampening=0.9,
                              weight_decay=args.wd)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=new_lr,
                               weight_decay=args.wd)
    else:
        raise Exception('Not supported optimizer: {0}'.format(args.optimizer))
    return optimizer

if __name__ == '__main__':
    main()