import numpy as np
class ToyModel:

    def __init__(self,
        wave,
        line=6_562.8,
        delta=10,
        cube=True
        ):

        self.wave = wave
        self.line = line
        self.delta = delta
        self.image = image

    def predict(self, image):

        if image.ndim == 3:
            image = image.reshape((1,) + image.shape)

        if not cube:
            return np.sum(image, axis=(1, 2, 3))


        neighborhood = (self.line-self.delta) < self.wave
        neighborhood *= self.wave < (self.line + self.delta)

        return np.sum(image[..., neighborhood], axis=(1, 2, 3))
