import numpy as np

import random
from fnmatch import fnmatch
from os import listdir
from os.path import join

class NdarrayDataset:
    """Loads data saved as ndarrays in .npy files using np.save(...).
    
    ndarray dimensions: T x H x W x D
    output: {'image': C x H x W x D, 'label': C x H x W x D}
        T: type (0: image, 1: label)
        C: channel
        H, W, D: spatial dimensions
    """
    def __init__(self, dir, transform = None):
        self.dir = dir
        self.files = [f for f in listdir(dir) if fnmatch(f, '*.npy')]
        self.transform = transform
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, i):
        return self.load(join(self.dir, self.files[i]))
        
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
    
    def load(self, filename):
        x, y = np.load(filename)
        example = {'image': x, 'label': y}
        if self.transform:
            example = self.transform(example)
        if len(example['image'].shape) == 3:
            x, y = example['image'], example['label']
            example = {'image': x[None,:,:,:], 
                       'label': y[None,:,:,:]}
        return example
    
    def shuffle(self):
        random.shuffle(self.files)

class NdarrayDatasetSplit(NdarrayDataset):
    """Splits the arrays somehow (abstract class)."""
    def __init__(self, dir, transform = None):
        super().__init__(dir, transform)
        self.i = 0
        self.example = super().__getitem__(self.i)
        self.depth = None
    
    def __len__(self):
        return super().__len__() * self.depth
    
    def __getitem__(self, i):
        i, d = i // self.depth, i % self.depth
        if i != self.i:
            self.i = i
            self.example = super().__getitem__(self.i)
        return self.pick(d)
    
    def pick(self, d):
        raise NotImplementedError
        
class NdarrayDataset2d(NdarrayDatasetSplit):
    """Splits the arrays by the D dimension."""
    def __init__(self, dir, transform = None):
        super().__init__(dir, transform)
        self.depth = self.example['image'].shape[3]
    
    def pick(self, d):
        x, y = self.example['image'], self.example['label']
        return {'image': x[:,:,:,d], 'label': y[:,:,:,d]}

class NdarrayDatasetPatch(NdarrayDatasetSplit):
    """Splits the arrays into patches along spatial dimensions.
    
    Patches have 1/2 overlap.
    """
    def __init__(self, dir, transform = None, patch_R = 8):
        super().__init__(dir, transform)
        self.patch_R = patch_R
        self.depth = patch_R ** 3
        self.size = (np.array(self.example['image'].shape[1:4]) 
            // patch_R)
    
    def pick(self, d):
        """d is a base (patch_R * (patch_R - 1)) number with
        length 3. Each digit is the starting point of the patch
        in each dimension.
        """
        x, y = self.example['image'], self.example['label']
        d1, d = d % self.patch_R, d // self.patch_R
        d2, d3 = d % self.patch_R, d // self.patch_R
        d1 *= self.size[0]
        d2 *= self.size[1]
        d3 *= self.size[2]
        slice = np.index_exp[:,d1:d1+self.size[0],
                             d2:d2+self.size[1],
                             d3:d3+self.size[2]]
        return {'image': x[slice], 'label': y[slice]}
