import nibabel as nib
import numpy as np

import random
from fnmatch import fnmatch
from os import listdir
from os.path import join

from transform import RealImag

class NdarrayDataset:
    """Loads data saved as ndarrays in .npy files using np.save(...).
    
    ndarray dimensions: T x H x W x D
    output: {'image': C x H x W x D, 'label': C x H x W x D}
        T: type (0: image, 1: label)
        C: channel
        H, W, D: spatial dimensions
    """
    def __init__(self, dir, transform = None, read = np.load):
        self.dir = dir
        self.files = [f for f in listdir(dir) 
                      if (fnmatch(f, '*.npy') or fnmatch(f, '*.nii'))]
        self.transform = transform
        self.read = read
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, i):
        return self.load(join(self.dir, self.files[i]))
        
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
    
    def load(self, filename):
        x, y = self.read(filename)
        example = {'image': x, 'label': y}
        if self.transform:
            example = self.transform(example)
        else:
            example = RealImag()(example)
        return example
    
    def shuffle(self):
        random.shuffle(self.files)

class NdarrayDatasetSplit(NdarrayDataset):
    """Splits the arrays somehow (abstract class)."""
    def __init__(self, dir, transform = None, read = np.load):
        super().__init__(dir, transform = transform, read = read)
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
        x, y = self.example['image'], self.example['label']
        s = self.slice(d)
        return {'image': x[s], 'label': y[s]}
       
    def slice(self, d):
        raise NotImplementedError
        
class NiiDataset2d(NdarrayDatasetSplit):
    """Splits the array by both D and E dimensions."""
    def __init__(self, dir, transform = None):
        def read(filename):
            slash = filename.rfind('/')
            img = (filename[:slash] + "/image" + filename[slash:-4] + 
                   '_M' + filename[-4:])
            image = nib.load(img).get_data().__array__()
            label = nib.load(filename).get_data().__array__()
            return image, label
        super().__init__(dir, transform = transform, read = read)
        self.d = self.example['image'].shape[3]
        self.e = self.example['image'].shape[4]
        self.depth = (self.d * self.e)
    
    def slice(self, d):
        dd, de = d % self.d, d // self.d
        return np.index_exp[:,:,:,dd,de]

class NdarrayDataset2d(NdarrayDatasetSplit):
    """Splits the arrays by the D dimension."""
    def __init__(self, dir, transform = None):
        super().__init__(dir, transform = transform)
        self.depth = self.example['image'].shape[3]
    
    def slice(self, d):
        return np.index_exp[:,:,:,d]

class NdarrayDatasetPatch(NdarrayDatasetSplit):
    """Splits the arrays into patches along spatial dimensions.
    
    Patches have 1/2 overlap.
    """
    def __init__(self, dir, transform = None, patch_R = 8):
        super().__init__(dir, transform = transform)
        self.patch_R = patch_R
        self.depth = patch_R ** 3
        self.size = (np.array(self.example['image'].shape[1:4]) 
            // patch_R)
    
    def slice(self, d):
        """d is a base (patch_R * (patch_R - 1)) number with
        length 3. Each digit is the starting point of the patch
        in each dimension.
        """
        d1, d = d % self.patch_R, d // self.patch_R
        d2, d3 = d % self.patch_R, d // self.patch_R
        d1 *= self.size[0]
        d2 *= self.size[1]
        d3 *= self.size[2]
        return np.index_exp[:,d1:d1+self.size[0],
                            d2:d2+self.size[1],
                            d3:d3+self.size[2]]
