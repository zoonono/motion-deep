import numpy as np
from scipy.signal import decimate
from scipy.ndimage import zoom
import torch

"""Brain data should be saved as complex ndarrays (H x W x D) using 
np.save(...). After applying transforms, the data should be torch
tensors in the form (C x H x W x D).
image and label should always be the same dimensions.
"""
class Transforms(object):
    """Composes one-argument transforms. Does not work for two-argument
    transforms like Residual."""
    def __init__(self, transforms_list, apply_to = 'both'):
        """apply_to: ['both', 'image', 'label]"""
        self.transforms_list = transforms_list
        self.apply_to = apply_to
        if apply_to not in ['both', 'image', 'label']:
            assert False, 'Option ' + apply_to + 'not recognized'
    
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        for t in self.transforms_list:
            if self.apply_to in ['both', 'image']:
                image = t(image)
            if self.apply_to in ['both', 'label']: 
                label = t(label)
        return {'image': image, 'label': label}

class MagPhase(object):
    """Splits complex data into mag and phase components as channels.
    (spatial dims) -> C x (spatial dims)
    Input arrays should be ndarrays.
    """
    def __call__(self, arr):
        arr = np.expand_dims(arr, axis = 0)
        arr = np.concatenate((np.abs(arr), np.angle(arr)), axis = 0)
        return arr

class RealImag(object):
    """Splits complex data into real and imag components as channels.
    (spatial dims) -> C x (spatial dims)
    Input arrays should be ndarrays.
    """
    def __call__(self, arr):
        arr = np.expand_dims(arr, axis = 0)
        arr = np.concatenate((np.real(arr), np.imag(arr)), axis = 0)
        return arr
    
class PickChannel(object):
    """C x (spatial dims) -> C x (spatial dims)
    Input arrays should be ndarrays.
    """
    def __init__(self, channel):
        self.channel = channel
        
    def __call__(self, arr):
        arr = arr[self.channel]
        arr = np.expand_dims(arr, axis = 0)        
        return arr

class Resize(object):
    def __init__(self, new_size):
        self.new_size = new_size
        
    def __call__(self, arr):
        """Resizes image to be new_size"""
        factor = np.array(self.new_size) / np.array(arr.shape)
        return zoom(arr, factor)

class Decimate(object):
    """Downsample axes in array by some factor.
    Input arrays should be ndarrays.
    """
    def __init__(self, factor = 2, axes = None):
        self.factor = factor
        self.axes = axes # if None, decimate all axes

    def __call__(self, arr):
        """Downsamples array by 2."""
        if self.axes:
            for axis in self.axes:
                arr = decimate(arr, self.factor, axis = axis)
        else: # if there is no specified list of axes, decimate spatial axes
            for axis in range(1, len(arr.shape)):
                arr = decimate(arr, self.factor, axis = axis)
        return arr
                
class Residual(object):
    """Predicts the residual instead of the label itself."""
    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        return {'image': image,
                'label': image - label}
                
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        return {'image': torch.from_numpy(image.copy()).float(),
                'label': torch.from_numpy(label.copy()).float()}
