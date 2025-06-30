# mri_augmentations.py

import numpy as np
import scipy.ndimage as ndi

class SimulateIntensityStriping:
    def __init__(self, prob=0.3, axis=2, frequency=20, amplitude=0.1, pattern='sin'):
        self.prob = prob
        self.axis = axis
        self.frequency = frequency
        self.amplitude = amplitude
        self.pattern = pattern

    def __call__(self, data_dict):
        if np.random.rand() < self.prob:
            img = data_dict["data"]
            for c in range(img.shape[0]):
                shape = img.shape[1:]  # (X, Y, Z)
                coords = np.arange(shape[self.axis])
                if self.pattern == 'sin':
                    stripe_pattern = np.sin(2 * np.pi * coords / self.frequency)
                else:
                    stripe_pattern = np.sign(np.sin(2 * np.pi * coords / self.frequency))

                # expand dims for broadcasting
                shape_expand = np.ones(len(shape), dtype=int)
                shape_expand[self.axis] = shape[self.axis]
                stripe_pattern = stripe_pattern.reshape(shape_expand)

                img[c] = img[c] * (1 + self.amplitude * stripe_pattern)
            data_dict["data"] = img
        return data_dict


class DownsampleUpsampleZ:
    def __init__(self, prob=0.3, factor=5, exclude_channels=[1]):
        """
        exclude_channels: list of channel indices NOT to apply downsampling to
        """
        self.prob = prob
        self.factor = factor
        self.exclude_channels = exclude_channels or []

    def __call__(self, data_dict):
        if np.random.rand() < self.prob:
            img = data_dict["data"]
            for c in range(img.shape[0]):
                if c in self.exclude_channels:
                    continue
                zoom_down = (1, 1, 1 / self.factor)
                zoom_up = (1, 1, self.factor)
                lowres = ndi.zoom(img[c], zoom_down, order=1)
                img[c] = ndi.zoom(lowres, zoom_up, order=1)
            data_dict["data"] = img
        return data_dict



class RandomGamma:
    def __init__(self, prob=0.3, gamma_range=(0.3, 1.7)):
        self.prob = prob
        self.gamma_range = gamma_range

    def __call__(self, data_dict):
        if np.random.rand() < self.prob:
            img = data_dict["data"]
            img_min = img.min()
            img_max = img.max()
            img_norm = (img - img_min) / (img_max - img_min + 1e-8)
            gamma = np.random.uniform(*self.gamma_range)
            img_gamma = img_norm ** gamma
            img = img_gamma * (img_max - img_min) + img_min
            data_dict["data"] = img
        return data_dict
