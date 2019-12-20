import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument("-c", "--config", help="name of config file. either MNIST_NO, MNIST_ATT, DD_MNIST_ATT", required=True, dest="config")
args = argparser.parse_args()
conf = args.config
# TODO add decent config support and other params
if conf == "MNSIT_NO":
    import configs.mnist_no_attention_config as config
elif conf == "MNIST_ATT":
    import configs.mnist_attention_config as config
elif conf == "DD_MNIST_ATT":
    import configs.dd_mnist_attention_config as config
else:
    import configs.dummy_config as config

# TODO parse args

NEED_EVAL = True
HAS_CUDA = True

import train as train_funcs
import model

dataset = config.dataset
imsize = config.imsize
save_prefix = config.save_prefix

HIDDEN = config.HIDDEN
Z = config.Z
T = config.T
READ_SIZE = config.READ_SIZE
WRITE_SIZE = config.WRITE_SIZE

lr = config.lr
lr_step = config.lr_step
lr_gamma = config.lr_gamma
BATCH_SIZE = config.BATCH_SIZE
EPOCHS = config.EPOCHS
grad_clip = config.grad_clip


model = model.DRAW(HIDDEN, Z, T, imsize[0], imsize[1], read_size=READ_SIZE, write_size=WRITE_SIZE)
if HAS_CUDA:
    model = model.cuda()

opt = optim.Adam(model.parameters(), lr, betas=(0.9, 0.99))
scheduler = optim.lr_scheduler.StepLR(opt, 5, 0.3)

train = DataLoader(dataset(mode="train"),
                   shuffle=True, batch_size=BATCH_SIZE)
test = DataLoader(dataset(mode="test"),
                  shuffle=True, batch_size=128)

train_funcs.train_model(model, opt, train, test, scheduler, EPOCHS, grad_clip=grad_clip, HAS_CUDA=HAS_CUDA, save_plot=True, save_prefix=save_prefix)
