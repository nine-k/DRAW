# params from the DRAW paper (except for learning params)
from binarized_mnist import BinarizedMNIST

dataset = BinarizedMNIST
imsize = (28, 28)
save_prefix = "MNIST_NO_ATT_"

# model params
T = 12
HIDDEN = 256
Z = 100
READ_SIZE = None
WRITE_SIZE = None

# learn params
lr = 1e-3
lr_step = 5
lr_gamma = 0.1
grad_clip = 4.
BATCH_SIZE = 64
EPOCHS = 20
