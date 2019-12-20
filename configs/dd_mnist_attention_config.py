from binarized_mnist_dd import BinarizedMNISTDoubleDigits

dataset = BinarizedMNISTDoubleDigits
imsize = (60, 60)
save_prefix = "DD_MNIST_ATT_"

# model params
T = 64
HIDDEN = 256
Z = 100
READ_SIZE = 5
WRITE_SIZE = 5

# learning params
lr = 2e-4
lr_step = 10
lr_gamma = 0.5
BATCH_SIZE = 128
EPOCHS = 40
grad_clip = 5.
