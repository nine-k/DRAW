from binarized_mnist import BinarizedMNIST
import numpy as np

class BinarizedMNISTDoubleDigit(BinarizedMnist):
    def __init__(self, canvas_size=(60, 60), **kwargs):
        super().__init__(**kwargs)
        self.canvas_size = canvas_size
    def __getitem__(self, ind):
        canvas = np.zeros(self.canvas_size, dtype=int)
        for _ in range(2):
            idx = np.random.randint(len(self))
            digit = np.array(self.data[idx])
            digit.resize(self.imsize)
            pos_x = np.random.randint(self.canvas_size[0] - self.imsize[0])
            pos_y = np.random.randint(self.canvas_size[1] - self.imsize[1])
            canvas[pos_x:pos_x + self.imsize[0], pos_y:pos_y + self.imsize[1]] += digit
        mask = canvas > 1
        canvas[mask] = 1
        canvas.resize(self.canvas_size[0]**2)
        return canvas

