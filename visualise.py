import argparse
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
#TODO add parse
model_path = "./MNIST_ATT_10"
from binarized_mnist import BinarizedMNIST
dataset = BinarizedMNIST
NUM_TO_VIS = 1
shape = (28, 28)
save_prefix = "./imgs/test_"

import torch
# model = torch.load(model_path)

def visualize_attention(x, y, delta, N, img):
    h, w = img.shape
    attention_mat = np.zeros((h, w))
    x_start = max(0, int(x - delta * N / 2.))
    y_start = max(0, int(y - delta * N / 2.))
    x_end = max(0, int(x + delta * N / 2.))
    y_end = max(0, int(y + delta * N / 2.))
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
    res_img = res_img.transpose(1, 2, 0)
    return res_img

model = torch.load(model_path)
data = next(iter(DataLoader(dataset(mode="test"),
                   shuffle=True, batch_size=NUM_TO_VIS))).float().cuda()
gen_history, attentions = model.forward(data, True)
data = data.cpu().numpy()
for idx in range(NUM_TO_VIS):
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
        fig.savefig("%s%d_%d.pdf" % (save_prefix, idx, t))
        plt.close("all")

