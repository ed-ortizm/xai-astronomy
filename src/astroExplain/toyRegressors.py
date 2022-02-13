import numpy as np
###############################################################################
class GalaxyPlus:
    """
    Class to add all pixel values in an image
    """

    def __init__(self):
        pass

    def predict(self, image: np.array) -> np.array:

        image = self._update_dimension(image)

        # predict and normalize
        prediction = np.sum(image, axis=(1, 2, 3))  # / image[0, :].size
        # print(prediction.shape)

        return prediction.reshape((-1, 1))

    def _update_dimension(self, image: np.array) -> np.array:

        if image.ndim == 3:
            return image[np.newaxis, ...]

        return image
###############################################################################
class CubePlus:

    def __init__(self,
        wave,
        line=6_562.8,
        delta=10,
        cube=True
        ):

        self.wave = wave
        self.line = line
        self.delta = delta
        self.cube = cube

    def predict(self, image):

        if image.ndim == 3:
            image = image.reshape((1,) + image.shape)

        if not self.cube:
            prediction = np.sum(image, axis=(1, 2, 3)).reshape((-1, 1))
            return prediction


        neighborhood = (self.line-self.delta) < self.wave
        neighborhood *= self.wave < (self.line + self.delta)

        prediction = np.sum(
            image[..., neighborhood],
            axis=(1, 2, 3)
            ).reshape((-1, 1))

        return prediction
