import urllib
import os
from numpy import array
from torch.utils.data import Dataset

class BinarizedMNIST(Dataset):
    URLS = {
        "train": "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_train.amat",
        "test": "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_test.amat",
        "dev":  "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_valid.amat",
    }
    def __init__(self, path="./", mode="train", load=True):
        if os.path.isdir(path):
            if not load:
                raise Exception("no such file")
            else:
                path = os.path.join(path, mode)
                urllib.request.urlretrieve(BinarizedMnist.URLS[mode], path)
        self.data = []
        with open(path, "r") as f:
            for line in f:
                tmp = line.split(' ')
                tmp = list(map(int, tmp))
                self.data.append(tmp)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):
        return array(self.data[ind])

