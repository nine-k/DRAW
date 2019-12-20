import numpy as np
from torchvision.datasets import MNIST
class BinarizedMNISTDoubleDigits(MNIST):
    def __init__(self, canvas_size=(60, 60), imsize=(28, 28), mode="train", load=True, path="./"):
        self.imsize = imsize
        super().__init__(path, train=(mode == "train"), download=load)
        self.canvas_size = canvas_size

    def __getitem__(self, ind):
        canvas = np.zeros(self.canvas_size, dtype=float)
        for _ in range(2):
            idx = np.random.randint(len(self))
            digit = np.array(super().__getitem__(idx)[0], dtype=float) / 255
            pos_x = np.random.randint(self.canvas_size[0] - self.imsize[0])
            pos_y = np.random.randint(self.canvas_size[1] - self.imsize[1])
            canvas[pos_x:pos_x + self.imsize[0], pos_y:pos_y + self.imsize[1]] += digit
        mask = canvas > 1
        canvas[mask] = 1
        canvas.resize(self.canvas_size[0]**2)
        return canvas
