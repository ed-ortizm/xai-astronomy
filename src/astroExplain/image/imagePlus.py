"""Toy model to compute the brightness of galaxy"""
import numpy as np


class GalaxyPlus:

    """
    Compute brightness of the image using as a baseline either
    the median or the mean value per channel.
    """

    def __init__(self, base_line: str = "median"):

        """
        INPUT
        base_line: either median or mean.
            brightness = image-[median or mean]
        """

        assert base_line in ("median", "mean")
        # assert (base_line == "median") or (base_line == "mean")

        self.base_line = base_line

    ###########################################################################
    def predict(self, image: np.array) -> np.array:
        """
        Compute normalized brightness of galaxy over base_line
        per channel

        INPUT
        image: 3D image or batch of 3D images

        OUTPUT
        predition: 2D array with brightness of images in the batch
            shape --> (batch_size, 1)
        """

        # (height, size, channels) -> (batch_size, height, size, channels)
        image = self._update_dimension(image)

        base_line_per_channel = self.get_base_line_per_channel(image)

        image = image - base_line_per_channel

        prediction = np.sum(image, axis=(1, 2, 3))

        return prediction.reshape((-1, 1))

    ###########################################################################
    def get_base_line_per_channel(self, image: np.array) -> np.array:
        """
        Compute baseline per channel, either mean or median

        INPUT
        image: 3D image or batch of 3D images

        OUTPUT
        array with median or mean per channel,
            keeping the dimensions of image
        """

        # axis=(1, 2) -> the weidth and height of the image
        if self.base_line == "median":

            return np.median(image, axis=(1, 2), keepdims=True)

        return np.mean(image, axis=(1, 2), keepdims=True)

    ###########################################################################
    @staticmethod
    def _update_dimension(image: np.array) -> np.array:

        if image.ndim == 3:
            return image[np.newaxis, ...]

        return image
