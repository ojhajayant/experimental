#!/usr/bin/env python
"""
lr_finder.py: This contains learning-rate-finder class definitions & utilities.
"""
from __future__ import print_function, with_statement, division
import copy
import os
import sys
import torch
from tqdm.autonotebook import tqdm
from torch.optim.lr_scheduler import _LRScheduler
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import display
import cfg
from utils import misc
from models import resnet18

sys.path.append('./')
global args
from cfg import get_args

args = get_args()
file_path = args.data

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
import matplotlib.pyplot as plt

class LRRangeFinder(_LRScheduler):
    def __init__(self, optimizer, init_lr, end_lr, num_epochs, alpha=0.3):
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.end_lr = end_lr
        self.num_epochs = num_epochs
        self.alpha = alpha
        self.best_loss = 1e9
        self.history = {"lr": [], "loss": []}

    def on_train_begin(self, logs=None):
        self.total_iterations = self.num_epochs * len(train_loader)
        self.iteration = 0
        self.lr = self.init_lr

    def on_batch_end(self, epoch, logs=None):
        current_loss = logs["loss"]
        if current_loss < self.best_loss:
            self.best_loss = current_loss
        lr = self.init_lr * (self.end_lr / self.init_lr) ** (self.iteration / self.total_iterations)
        self.history["lr"].append(lr)
        self.optimizer.param_groups[0]["lr"] = lr
        self.history["loss"].append(current_loss)
        self.iteration += 1

    def on_train_end(self, logs=None):
        lrs = np.array(self.history["lr"])
        losses = np.array(self.history["loss"])
        smoothed_losses = np.convolve(losses, np.ones(self.alpha) / self.alpha, mode="valid")
        derivative = np.gradient(smoothed_losses)
        idx = np.argmin(derivative)
        self.lr = lrs[idx]
        print(f"Found LR: {self.lr:.3e}")
        
    def plot(self, show_lr=None, yaxis_label="Loss"):
        plt.plot(self.history["lr"], self.history["loss"])
        if show_lr is not None:
            plt.axvline(x=show_lr, color="red")
        plt.xscale("log")
        plt.xlabel("Learning rate")
        plt.ylabel(yaxis_label)
        plt.savefig('plot13.png')
            from IPython.display import Image
            Image(filename='plot13.png')
            display(plt.gcf())
        plt.show()

def find_network_lr(model, criterion, optimizer, device, train_loader, init_lr, init_weight_decay, end_lr=1, num_epochs=100, L1=False):
    if L1:
        reg = 0
        for param in model.parameters():
            reg += torch.norm(param, 1)
    else:
        reg = 0
        for param in model.parameters():
            reg += torch.norm(param, 2)

    model.to(device)
    optimizer = optimizer(model.parameters(), lr=init_lr, weight_decay=init_weight_decay)
    lr_finder = LRRangeFinder(optimizer, init_lr, end_lr, num_epochs)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y) + reg
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.shape[0]
            lr_finder.on_batch_end(epoch, {"loss": loss.item()})
        train_loss /= len(train_loader.dataset)
    lr_finder.on_train_end()
    lr_finder.plot(show_lr=lr_finder.lr, yaxis_label="Training Accuracy")
    return lr_finder.lr
