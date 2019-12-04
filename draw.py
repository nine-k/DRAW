import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from collections import defaultdict

from model import DRAW
from binarized_mnist import BinarizedMNIST
from util import * #TODO make explicit

HAS_CUDA = torch.cuda.device_count() > 0
if HAS_CUDA:
    print("GPU available")
else:
    print("No GPU available")
imsize = (28, 28)

def calc_metrics(pred, x):
    x = x.cpu().numpy()
    pred = pred.cpu().numpy() > 0.5
    TP = ((pred == 1) & (x == 1)).sum()
    FP = ((pred == 1) & (x == 0)).sum()
    FN = ((pred == 0) & (x == 1)).sum()
    TN = ((pred == 0) & (x == 0)).sum()

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


T = 12
HIDDEN = 256
Z = 100
lr = 1 * 1e-3
grad_clip = 4.

BATCH_SIZE = 64
EPOCHS = 20

model = model.DRAW(HIDDEN, Z, T, imsize[0], imsize[1]).cuda()
opt = optim.Adam(model.parameters(), lr, betas=(0.5, 0.99))
scheduler = optim.lr_scheduler.StepLR(opt, 5, 0.3)

train = DataLoader(BinarizedMNIST(mode="train"),
                   shuffle=True, batch_size=BATCH_SIZE)
test = DataLoader(BinarizedMNIST(mode="test"),
                  shuffle=True, batch_size=128)

reg_coeff=1.
train_history = defaultdict(list)
test_history = defaultdict(list)
generation_history = []

#TODO move to functions
for epoch in range(EPOCHS):
    print("epoch number: %d" % epoch)
    print("lr: %s", scheduler.get_lr())
    model.train()
    for x in tqdm(train):
        x = x.float()
        if HAS_CUDA:
            x = x.cuda()
        opt.zero_grad()
        pred = model.forward(x)
        lx, lz = model.loss(pred, x)
        loss = lx + reg_coeff * lz
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        p, r, f1 = calc_metrics(pred.detach(), x)
        train_history["Lx"].append(lx.item())
        train_history["Lz"].append(lz.item())
        train_history["precision"].append(p)
        train_history["recall"].append(r)
        train_history["f1"].append(f1)
    model.eval()
    last_batch = None
    for x in test:
        lxs = []
        lzs = []
        ps = []
        rs = []
        f1s = []
        x = x.float()
        with torch.no_grad():
            if HAS_CUDA:
                x = x.cuda()
            pred = model.forward(x)
            lx, lz = model.loss(pred, x)
            p, r, f1 = calc_metrics(pred, x)
            ps.append(p)
            rs.append(r)
            f1s.append(f1)
            lxs.append(lx.item())
            lzs.append(lz.item())
            last_batch = pred
    test_history["Lx"].append(np.mean(lxs))
    test_history["Lz"].append(np.mean(lzs))
    test_history["precision"].append(np.mean(p))
    test_history["recall"].append(np.mean(r))
    test_history["f1"].append(np.mean(f1))
    scheduler.step()
    print("eval NATs: %f" % (test_history["Lx"][-1] + test_history["Lz"][-1]))
    plot_history(train_history, test_history)
    generation_history.append(unflatten(model.generate(1), imsize).cpu().detach())
    show_imgs((
        unflatten(torch.sigmoid(model.c_0.cpu().detach()), imsize),
        generation_history[-1]
    ), 2)

ims = unflatten(torch.cat(model.generate(10, True)), imsize).cpu().detach()

show_imgs(ims, 10, figsize=(10, 12))

