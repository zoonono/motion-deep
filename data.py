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

class Splitter(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset # C x H x W x D
        self.depth = dataset[0]['image'].shape[3]
        
    def __len__(self):
        return len(self.dataset) * self.depth
        
    def __getitem__(self, i):
        i, r = i // depth, i % depth
        example = self.dataset[i]
        x, y = example['image'], example['label']
        return {'image': x[:,:,:,r], 'label': y[:,:,:,r]}
        
    def __iter__(self):
        for example in self.dataset:
            x, y = example['image'], example['label']
            for i in self.depth:
                yield {'image': x[:,:,:,i], 'label': y[:,:,:,i]}
            
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

class RemoveDims(object):
    """Removes singleton dimensions.
    """
    
    def __init__(self, axes = None, both = False):
        self.axes = axes
        self.both = both
        
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        image = np.squeeze(image, axis = self.axes)
        if self.both:
            label = np.squeeze(label, axis = self.axes)
        return {'image': image,
                'label': label}
                
class FrontDim(object):
    """Adds a singleton dimension at the front of a numpy array.
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = np.expand_dims(image, axis = 0)
        label = np.expand_dims(label, axis = 0)
        
        return {'image': image,
                'label': label}

class BackDim(object):
    """Adds a singleton dimension at the end of a numpy array.
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = np.expand_dims(image, axis = len(image.shape))
        label = np.expand_dims(label, axis = len(label.shape))
        
        return {'image': image,
                'label': label}
        
class Transpose3d(object):
    """Transpose D x H x W x C to C x D x H x W.
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # numpy image: H x W x D x C
        # torch image: C x H x W x D
        image = image.transpose((3, 0, 1, 2))
        label = label.transpose((3, 0, 1, 2))
        return {'image': image,
                'label': label}
