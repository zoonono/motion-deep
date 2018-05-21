import numpy as np
from scipy.signal import decimate
from scipy.ndimage import zoom
import torch

"""Brain data should be saved as complex ndarrays (H x W x D) using 
np.save(...). After applying transforms, the data should be torch
tensors in the form (C x H x W x D).
image and label should always be the same dimensions.
"""

class Resize(object):
    def __init__(self, new_size):
        self.new_size = new_size
        
    def resize(self, arr):
        """Resizes image to be new_size"""
        factor = np.array(self.new_size) / np.array(arr.shape)
        return zoom(arr, factor)
        
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = self.resize(image)
        
        # only resize image because this will only be used for PD, which has no label
        return {'image': image,
                'label': image}

class Decimate(object):
    """Downsample axes in array by some factor.
    Input arrays should be ndarrays.
    """
    def __init__(self, factor = 2, axes = None):
        self.factor = factor
        self.axes = axes # if None, decimate all axes

    def downsample(self, arr):
        """Downsamples array by 2."""
        if self.axes:
            for axis in self.axes:
                arr = decimate(arr, self.factor, axis = axis)
        else: # if there is no specified list of axes, decimate spatial axes
            for axis in range(1, len(arr.shape)):
                arr = decimate(arr, self.factor, axis = axis)
        return arr

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        return {'image': self.downsample(image),
                'label': self.downsample(label)}
                
class Residual(object):
    """Predicts the residual instead of the label itself."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        return {'image': image,
                'label': image - label}
                
class RealImag(object):
    """Splits complex data into real and imag components as channels.
    (spatial dims) -> C x (spatial dims)
    Input arrays should be ndarrays.
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = np.expand_dims(image, axis = 0)
        label = np.expand_dims(label, axis = 0)
        image = np.concatenate((np.real(image), np.imag(image)), 
            axis = 0)
        label = np.concatenate((np.real(label), np.imag(label)), 
            axis = 0)

        return {'image': image,
                'label': label}

class MagPhase(object):
    """Splits complex data into mag and phase components as channels.
    (spatial dims) -> C x (spatial dims)
    Input arrays should be ndarrays.
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = np.expand_dims(image, axis = 0)
        label = np.expand_dims(label, axis = 0)
        image = np.concatenate((np.abs(image), np.angle(image)), 
            axis = 0)
        label = np.concatenate((np.abs(label), np.angle(label)), 
            axis = 0)

        return {'image': image,
                'label': label}           
                
class PickChannel(object):
    """C x (spatial dims) -> C x (spatial dims)
    Input arrays should be ndarrays.
    """
    def __init__(self, channel):
        self.channel = channel
        
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image, label = image[self.channel], label[self.channel]
        image = np.expand_dims(image, axis = 0)
        label = np.expand_dims(label, axis = 0)
        
        return {'image': image,
                'label': label}
                
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        return {'image': torch.from_numpy(image.copy()).float(),
                'label': torch.from_numpy(label.copy()).float()}
