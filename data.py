import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from scipy.signal import decimate
import time
from random import shuffle

class GenericFilenames:
    def __init__(self, stem, image, label, ext, size, offset = 0):
        self.stem = stem
        self.image = image
        self.label = label
        self.ext = ext
        self.size = size
        self.offset = offset

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        x_path = self.stem + self.image + str(i + self.offset) + self.ext
        y_path = self.stem + self.label + str(i + self.offset) + self.ext
        return x_path, y_path

    def split(self, proportions):
        splitted_filenames = []
        offset = 0
        for p in proportions:
            splitted_filenames.append(GenericFilenames(self.stem, self.image,
                    self.label, self.ext, int(self.size * p), offset = offset))
            offset += int(self.size * p)
        return splitted_filenames
    
    def __iter__(self):
        i = 0
        while i < len(self):
            yield self[i]
            i += 1

class MotionCorrDataset(Dataset):
    def __init__(self, filenames, load_func, transform = None):
        self.filenames = list(filenames)
        self.load_func = load_func
        self.transform = transform

    def __len__(self):
        return (len(self.filenames))
    
    def load(self, filename):
        x_path, y_path = filename
        x = self.load_func(x_path)
        y = self.load_func(y_path)
        sample = {'image': x, 'label': y}
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def __getitem__(self, i):
        return self.load(self.filenames[i])
    
    def __iter__(self):
        for filename in self.filenames:
            yield self.load(filename)
    
    def shuffle(self):
        shuffle(self.filenames)

class Decimate(object):
    """Undersample each axis by some factor."""
    def __init__(self, factor = 2, axes = None):
        self.factor = factor
        self.axes = axes # if None, decimate all axes

    def downsample(self, arr):
        """Downsamples axes in array by factor of 2 using scipy's decimate."""
        if self.axes:
            for axis in self.axes:
                arr = decimate(arr, self.factor, axis = axis)
        else: # if there is no specified list of axes, decimate all axes
            for axis in range(len(arr.shape)):
                arr = decimate(arr, self.factor, axis = axis)
        return arr

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        return {'image': self.downsample(image),
                'label': self.downsample(label)}

class Residual(object):
    """Saves the residual (image - label) instead of the label itself."""

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

class TransposeBack(object):
    """B x C x H x W x D -> H x W x D"""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        return {'image': image[0,0,:,:,:],
                'label': label}
                
class BatchDim(object):
    """C x H x W x D -> B x C x H x W x D 
    or C x H x W -> B x C x H x W
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        dims = len(image.shape)
        if dims == 3:
            dict = {'image': image[None,:,:,:],
                    'label': label[None,:,:,:]}
        elif dims == 4:
            dict = {'image': image[None,:,:,:,:],
                    'label': label[None,:,:,:,:]}
        else:
            assert False, "Image must be 3d or 4d, but it is " + dims + "d"
        return dict

class DepthDim(object):
    """H x W -> H x W x D,
    C x H x W -> C x H x W x D
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        dims = len(image.shape)
        if dims == 2:
            dict = {'image': image[:,:,None],
                    'label': label}
        elif dims == 3:
            dict = {'image': image[:,:,:,None],
                    'label': label}
        else:
            assert False, "Image must be 2d or 3d, but it is " + dims + "d"
        return dict
        
class DepthDim2(object):
    """H x W -> H x W x D,
    C x H x W -> C x H x W x D
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        dims = len(image.shape)
        if dims == 2:
            dict = {'image': image[:,:,None],
                    'label': label[:,:,None]}
        elif dims == 3:
            dict = {'image': image[:,:,:,None],
                    'label': label[:,:,None]}
        else:
            assert False, "Image must be 2d or 3d, but it is " + dims + "d"
        return dict
        
class RemoveDim(object):
    """C x H x W x D <- B x C x H x W x D 
    or C x H x W <- B x C x H x W
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        dims = len(image.shape)
        if dims == 3:
            dict = {'image': image[0,:,:],
                    'label': label}
        elif dims == 4:
            dict = {'image': image[0,:,:,:],
                    'label': label}
        else:
            assert False, "Image must be 3d or 4d, but it is " + dims + "d"
        return dict
        
class Transpose2d(object):
    """Transpose ndarrays to C x H x W."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # numpy image: H x W x C
        # torch image: C x H x W
        if len(image.shape) == 2:
            image = image[:,:,None]
            label = label[:,:,None]
        image = image.transpose((2, 0, 1))
        label = label.transpose((2, 0, 1))
        return {'image': image,
                'label': label}
        
class Transpose3d(object):
    """Transpose ndarrays to C x D x H x W."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # numpy image: H x W x D x C
        # torch image: C x H x W x D
        if len(image.shape) == 3:
            image = image[:,:,:,None]
            label = label[:,:,:,None]
        image = image.transpose((3, 0, 1, 2))
        label = label.transpose((3, 0, 1, 2))
        return {'image': image,
                'label': label}
