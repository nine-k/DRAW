# params from the DRAW paper (except for learning params)
from binarized_mnist import BinarizedMNIST

dataset = BinarizedMNIST
imsize = (28, 28)
save_prefix = "MNIST_ATT_"

# model params
T = 64
HIDDEN = 256
Z = 100
READ_SIZE = 2
WRITE_SIZE = 5

# learning params
lr = 4e-4
lr_step = 10
lr_gamma = 0.6
BATCH_SIZE = 128
EPOCHS = 30
grad_clip = -4.

