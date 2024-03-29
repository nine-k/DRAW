import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict
from os import path
import numpy as np
from matplotlib import pyplot as plt
import util
from tqdm import tqdm

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


def append_dict(dest, to_add):
    for k in to_add.keys():
        dest[k].extend(to_add[k])
    return dest

def train_epoch(model, train, opt, grad_clip, HAS_CUDA=True):
    model.train()
    history = defaultdict(list)
    for x in tqdm(train):
        x = x.float()
        if HAS_CUDA:
            x = x.cuda()
        opt.zero_grad()
        pred = model.forward(x)
        lx, lz = model.loss(pred, x)
        loss = lx + lz
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        p, r, f1 = calc_metrics(pred.detach(), x)
        history["Lx"].append(lx.item())
        history["Lz"].append(lz.item())
        history["precision"].append(p)
        history["recall"].append(r)
        history["f1"].append(f1)
    return history

def test_epoch(model, test, HAS_CUDA=True):
    history = defaultdict(list)
    model.eval()
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
            history["Lx"].append(lx.item())
            history["Lz"].append(lz.item())
            history["precision"].append(p)
            history["recall"].append(r)
            history["f1"].append(f1)
    for k in history.keys():
        history[k] = [np.mean(history[k])]
    return history

def train_model(model, opt, train, test, scheduler, epochs, grad_clip=-1, HAS_CUDA=True,
                need_test=True, show_plot=False, save_plot=False,
                save_dir=".", save_prefix=""):
    train_history = defaultdict(list)
    test_history = defaultdict(list)

    for epoch in range(epochs):
        print("starting epoch %d" % epoch)
        train_history = append_dict(train_history, train_epoch(model, train, opt, grad_clip, HAS_CUDA))
        if need_test:
            test_history = append_dict(test_history, test_epoch(model, test, HAS_CUDA))
            lx = test_history["Lx"][-1]
            lz = test_history["Lz"][-1]
            print("Eval NATS: %f" % (lx + lz))
            print("Eval reconstruction loss %f" % lx)
            print("Eval divergence loss %f" % lz)
            print("Eval f1: %f" % (test_history["f1"][-1]))
        draw_plot = show_plot or save_plot
        if draw_plot:
            plt.close('all')
            fig = util.plot_history(train_history, test_history, show_plot)
            if save_plot:
                fig.savefig(path.join(save_dir, save_prefix + 'plot.pdf'))
        if save_dir is not None:
            torch.save(model, path.join(save_dir, save_prefix + str(epoch)))
        scheduler.step()

