import numpy as np
from torch.utils.data import DataLoader

from util import unflatten
import imageio
from matplotlib import pyplot as plt

from os import path
import os

import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument("-m", "--model", help="path to model", required=True, dest="model")
argparser.add_argument("-g", "--gif", help="generate gif", required=False, action="store_false", dest="gif")
argparser.add_argument("-p", "--prefix", help="file_prefix", required=False, dest="prefix", default="vis_")
argparser.add_argument("-d", "--dataset", help="dataset name MNIST or MNIST_DD", required=True, dest="dataset")
args = argparser.parse_args()

model_path = args.model
need_gif = args.gif
save_prefix = args.prefix
dataset = args.dataset

print(need_gif)

if dataset == "MNIST":
    shape = (28, 28)
elif dataset == "DD_MNIST":
    shape = (60, 60)

NUM_TO_VIS = 1
save_dir = "./imgs"
img_format = "png"

import torch
model = torch.load(model_path)

imgs = model.generate(1, True)

images = [None] * len(imgs)
for idx, img in enumerate(imgs):
    img = unflatten(img, shape).squeeze().cpu().numpy()
    fig = plt.figure()
    plt.axis("off")
    plt.imshow(img, cmap='gray')
    p = "%s%d_read.%s" % (path.join(save_dir, save_prefix), idx, "png")
    fig.savefig(p)
    plt.close("all")
    if need_gif:
        images[idx] = imageio.imread(p)
if need_gif:
    print('./%s_generate.gif' % (save_prefix))
    imageio.mimsave('./%s_generate.gif' % (save_prefix), images)

