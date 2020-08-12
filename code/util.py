import torch
import os
import shutil
import time
from collections import OrderedDict
import json
import torch.optim as optim
import pandas as pd
from model import CovidNet
import csv
import numpy as np

best_train_acc = 0
best_val_acc = 0
best_test_acc = 0
best_acc = 0


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


def save_checkpoint(state, is_best, path, filename='last'):
    name = os.path.join(path, filename + '_checkpoint.pth')
    print(name, '\n')
    torch.save(state, name)


def save_model(model, optimizer, args, metrics, epoch, best_pred_loss, confusion_matrix):
    global best_acc
    loss = metrics.data['loss']
    save_path = args.save
    mkdir(save_path)

    with open(save_path + '/training_arguments.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    is_best = False
    if best_acc < metrics.data['correct'] / metrics.data['total']:
        is_best = True
        best_pred_loss = loss
        best_acc = metrics.data['correct'] / metrics.data['total']
        save_checkpoint({'state_dict': model.state_dict(),
                         'metrics': metrics.data},
                        is_best, save_path, "best")
    else:
        save_checkpoint({'state_dict': model.state_dict(),
                         'metrics': metrics.data},
                        is_best, save_path, "last")

    return best_pred_loss


class Metrics:
    def __init__(self, path, keys=None, writer=None):
        self.writer = writer

        self.data = {'correct': 0,
                     'total': 0,
                     'loss': 0,
                     'accuracy': 0,
                     'top1_correct': 0,
                     'top3_correct': 0
                     }
        self.save_path = path

    def reset(self):
        for key in self.data:
            self.data[key] = 0

    def update_key(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self.data[key] += value

    def update(self, values):
        for key in self.data:
            self.data[key] += values[key]

    def avg_acc(self):
        return self.data['correct'] / self.data['total']

    def avg_loss(self):
        return self.data['loss'] / (self.data['total'] / 32.)

    def save(self):
        with open(self.save_path, 'w') as save_file:
            a = 0  # csv.writer()
            # TODO


def print_summary(args, epoch, metrics, mode=''):
    global best_train_acc
    global best_test_acc
    global best_val_acc
    if mode == 'train':
        if best_train_acc < 100. * metrics.data['correct'] / metrics.data['total']:
            best_train_acc = 100. * metrics.data['correct'] / metrics.data['total']
        print(
            mode + "\tEPOCH:{:2d}/{:3d}\tCorrect:{:5d}/{:5d}\t\tLoss:{:.6f}\tAcc:{:.2f}%\tBest Acc:{:.2f}%\n".format(
                epoch, args.nEpochs,
                metrics.data['correct'],
                metrics.data['total'],
                metrics.data['loss'] / (metrics.data['total'] // args.batch_size + 1),
                100. * metrics.data['correct'] / metrics.data['total'],
                best_train_acc
            ))
    if mode == 'test':
        if best_test_acc < 100. * metrics.data['correct'] / metrics.data['total']:
            best_test_acc = 100. * metrics.data['correct'] / metrics.data['total']
        print(
            mode + "\tEPOCH:{:2d}/{:3d}\tCorrect:{:5d}/{:5d}\t\tLoss:{:.6f}\tAcc:{:.2f}%\tBest Acc:{:.2f}%\n".format(
                epoch, args.nEpochs,
                metrics.data['correct'],
                metrics.data['total'],
                metrics.data['loss'] / (metrics.data['total'] // args.batch_size + 1),
                100. * metrics.data['correct'] / metrics.data['total'],
                best_test_acc
            ))


def read_filepaths(file, mode):
    paths, labels = [], []
    with open(file, 'r') as f:
        lines = f.read().splitlines()

        for idx, line in enumerate(lines):
            if '/ c o' in line:
                break
            try:
                subjid, path1, label = line.split(' ')
                path = '../data/COVID-CT/' + mode + '/' + path1
            except:
                print(line)

            paths.append(path)
            labels.append(label)
    return paths, labels


def read_filepaths2(file):
    paths, labels = [], []
    with open(file, 'r') as f:
        lines = f.read().splitlines()

        for idx, line in enumerate(lines):
            if '/ c o' in line:
                break
            try:
                subjid, path1, path2, label = line.split(' ')
                path = path1 + ' ' + path2
            except:
                print(line)

            paths.append(path)
            labels.append(label)
    return paths, labels
