import argparse
import numpy as np
#TODO add parse
model_path = ""

# import torch
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
    return res_img
