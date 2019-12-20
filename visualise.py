import numpy as np
from torch.utils.data import DataLoader

import imageio
from matplotlib import pyplot as plt

from os import path
import os

import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument("-m", "--model", help="path to model", required=True, dest="model")
argparser.add_argument("-d", "--dataset", help="dataset name MNIST or MNIST_DD", required=True, dest="dataset")
argparser.add_argument("-g", "--gif", help="generate gif", required=False, action="store_false", dest="gif")
argparser.add_argument("-p", "--prefix", help="file_prefix", required=False, dest="prefix", default="vis_")
args = argparser.parse_args()

model_path = args.model
dataset = args.dataset
need_gif = args.gif
save_prefix = args.prefix

print(need_gif)

if dataset == "MNIST":
    from binarized_mnist import BinarizedMNIST as dataset
    shape = (28, 28)
elif dataset == "MNIST_DD":
    from binarized_mnist_dd import BinarizedMNISTDoubleDigit as dataset
    shape = (60, 60)
else:
    raise ValueError("no such dataset")

NUM_TO_VIS = 1
save_dir = "./imgs"
img_format = "png"

import torch
# model = torch.load(model_path)

def visualize_attention(x, y, delta, N, img):
    h, w = img.shape
    attention_mat = np.zeros((h, w))
    x_start = max(0, int(x - delta * (N - 1) / 2.))
    y_start = max(0, int(y - delta * (N - 1) / 2.))
    x_end = max(0, int(x + delta * (N - 1) / 2.))
    y_end = max(0, int(y + delta * (N - 1) / 2.))
    attention_mat[y_start : y_end, x_start : x_end] = 1. # make full square
    if y_end - y_start > 2:
        y_end -= 1
        y_start += 1
    if x_end - x_start > 2:
        x_end -= 1
        x_start += 1
    attention_mat[y_start : y_end, x_start : x_end] = 0. # leave contour
    res_img = np.stack([img] * 3)
    res_img[0] += attention_mat
    res_img[1, attention_mat == 1] = 0
    res_img[2, attention_mat == 1] = 0
    res_img = res_img.transpose(1, 2, 0)
    return res_img

model = torch.load(model_path)
data = next(iter(DataLoader(dataset(mode="test"),
                   shuffle=True, batch_size=NUM_TO_VIS))).float().cuda()
gen_history, attentions = model.forward(data, True)
data = data.cpu().numpy()
for idx in range(NUM_TO_VIS):
    att_images = []
    write_images = []
    for t in range(model.T):
        attention = attentions[t]
        hilighted = visualize_attention(attention["g_x"][idx],
                                       attention["g_y"][idx],
                                       attention["delta"][idx],
                                       model.read_size,
                                       data[idx].reshape(shape)
                                   )
        fig = plt.figure()
        plt.axis("off")
        plt.imshow(hilighted)
        p = "%s%d_%d_read.%s" % (path.join(save_dir, save_prefix), idx, t, img_format)
        fig.savefig(p)
        if need_gif:
            att_images.append(imageio.imread(p))

        fig = plt.figure()
        plt.axis("off")
        img = gen_history[t][idx]
        img.resize(*shape)
        plt.imshow(img, cmap="gray")
        p = "%s%d_%d_read.%s" % (path.join(save_dir, save_prefix), idx, t, img_format)
        fig.savefig(p)
        plt.close("all")
        if need_gif:
            write_images.append(imageio.imread(p))
    if need_gif:
        print('saving')
        imageio.mimsave('./%s%d_read.gif' % (save_prefix, idx), att_images)
        imageio.mimsave('./%s%d_write.gif' % (save_prefix, idx), write_images)

