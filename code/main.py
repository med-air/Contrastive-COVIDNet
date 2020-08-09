import warnings
warnings.filterwarnings('ignore')
import os
import argparse
import torch
import numpy as np
import util
from train import *
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import time
import math


def adjust_learning_rate(optimizer, epoch, args):
    lr_min = 0
    if args.cosine:
        lr = math.fabs(lr_min + (1 + math.cos(1 * epoch * math.pi / args.nEpochs)) * (args.lr - lr_min) / 2.)
    else:
        lr = args.lr * (0.1 ** (epoch // 30))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    args = get_arguments()
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

    util.mkdir('runs')

    test_acc_file = 'runs/test' + str(args.cosine) + str(args.bna) + str(args.bnd) + '.txt'
    train_acc_file = 'runs/train' + str(args.cosine) + str(args.bna) + str(args.bnd) + '.txt'
    open(test_acc_file, 'w')
    open(train_acc_file, 'w')

    print('Building model, loading data...\n')
    if args.cuda:
        torch.cuda.manual_seed(SEED)

    model, optimizer, training_generator, test_generator = initialize(args)

    best_pred_loss = 1000.0
    print('\nCheckpoint folder:', args.save,
          '\n\nCosine:', args.cosine, '\t\tBna:', args.bna, '\t\tBnd:', args.bnd, '\t\tContrastive:', args.cont,
          '\n\nStart training...\n')

    for epoch in range(1, args.nEpochs + 1):
        train_metrics = train(args, model, training_generator, optimizer, epoch)
        test_metrics, confusion_matrix, ucsd_correct_total, sars_correct_total, ucsd_test_total, sars_test_total \
            = validation(args, model, test_generator, epoch, mode='test')

        best_pred_loss = util.save_model(model, optimizer, args, test_metrics, epoch, best_pred_loss, confusion_matrix)

        print('COVID-CT Accuracy: {0:.2f}%\tSARS-Cov-2 Accuracy: {1:.2f}%\n'.format(
            100. * ucsd_correct_total / ucsd_test_total, 100. * sars_correct_total / sars_test_total))

        with open(test_acc_file, 'a+') as f:
            f.write(str(test_metrics.data['correct'] / test_metrics.data['total']) + ' ' +
                    str(optimizer.param_groups[0]['lr']) + ' ' +
                    str(test_metrics.data['loss'] / (test_metrics.data['total'] // args.batch_size + 1)) + '\n')
        with open(train_acc_file, 'a+') as f:
            f.write(str(train_metrics.data['correct'] / train_metrics.data['total']) + ' ' +
                    str(optimizer.param_groups[0]['lr']) + ' ' +
                    str(train_metrics.data['loss'] / (train_metrics.data['total'] // args.batch_size + 1)) + '\n')

        adjust_learning_rate(optimizer, epoch, args)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--nEpochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=643)
    parser.add_argument('--classes', type=int, default=2)
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', default=1e-7, type=float,
                        help='weight decay (default: 1e-7)')
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--dataset', type=str, default='../data',
                        help='path to dataset ')
    parser.add_argument('--save', type=str, default='saved/cont',
                        help='path to checkpoint ')
    parser.add_argument('--cosine', default=False,
                        help='learning rate adjust scheme ')
    parser.add_argument('--bnd', default=False,
                        help='batchnorm layers during feature extraction')
    parser.add_argument('--bna', default=False,
                        help='batchnorm layers after feature extraction')
    parser.add_argument('--cont', default=False,
                        help='contrastive objective')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    since1 = time.time()
    main()
    done1 = time.time()
    print('Total Time:', done1 - since1, 's\n')
