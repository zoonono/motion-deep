import nibabel as nib
import numpy as np

import random
from fnmatch import fnmatch
from os import listdir
from os.path import join

from motionsim import motion_PD
from transform import RealImag

class CombinedDataset:
    """Combines two datasets into one."""
    def __init__(self, d1, d2):
        self.d1 = d1
        self.d2 = d2
    
    def __len__(self):
        return len(self.d1) + len(self.d2)
    
    def shuffle(self):
        self.d1.shuffle()
        self.d2.shuffle()
    
    def __getitem__(self, i):
        if i < len(self.d1):
            return self.d1[i]
        return self.d2[i - len(self.d1)]
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
            
class NdarrayDataset:
    """Loads data saved as ndarrays in .npy files using np.save(...).
    
    ndarray dimensions: T x H x W x D
    output: {'image': C x H x W x D, 'label': C x H x W x D}
        T: type (0: image, 1: label)
        C: channel
        H, W, D: spatial dimensions
    """
    def __init__(self, dir, transform, read = np.load):
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
        
class Split():
    """Splits the arrays somehow (abstract class)."""
    def __init__(self, dataset):
        self.dataset = dataset
        self.i = 0
        self.example = dataset[self.i]
        self.depth = None
    
    def __len__(self):
        return len(self.dataset) * self.depth
    
    def __getitem__(self, i):
        i, d = i // self.depth, i % self.depth
        if i != self.i:
            self.i = i
            self.example = self.dataset[self.i]
        return self.pick(d)
    
    def pick(self, d):
        x, y = self.example['image'], self.example['label']
        s = self.slice(d)
        return {'image': x[s], 'label': y[s]}
       
    def slice(self, d):
        raise NotImplementedError

class Split4th(Split):
    """Splits the arrays by the 4th spatial dimension."""
    def __init__(self, dataset):
        super().__init__(dataset)
        self.depth = self.example['image'].shape[4]
    
    def slice(self, d):
        return np.index_exp[:,:,:,:,d]
    
class Split2d(Split):
    """Splits the arrays by the 3rd spatial dimension."""
    def __init__(self, dataset):
        super().__init__(dataset)
        self.depth = self.example['image'].shape[3]
    
    def slice(self, d):
        return np.index_exp[:,:,:,d]

class SplitPatch(Split):
    """Splits the arrays into patches along spatial dimensions.
    The arrays must be C x H x W x D
    """
    def __init__(self, dataset, patch_R = 8):
        super().__init__(dataset)
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

def r1(filename):
    """Loads nii files where the directory structure is:
    v dir
        v image
            brain1_M.nii
            brain2_M.nii
            ...
        brain1.nii
        brain2.nii
        ...
    """
    slash = filename.rfind('/')
    img = (filename[:slash] + "/image" + filename[slash:-4] + 
           '_M' + filename[-4:])
    image = nib.load(img).get_data().__array__()
    label = nib.load(filename).get_data().__array__()
    return image, label

def r2(filename):
    """Loads nii files where the directory structure is:
    v dir
        brain1.nii
        brain2.nii
        ...
    Used for PD data where there is no label.
    """
    image = nib.load(filename).get_data().__array__()
    return image, None

def r3(filename):
    """Loads nii files where the directory structure is:
    v dir
        brain1.nii
        brain2.nii
        ...
    Uses motion simulation to create image-label pairs during training.
    """
    label = nib.load(filename).get_data().__array__()
    image = np.zeros(label.shape, dtype = np.complex64)
    for e in range(image.shape[3]):
        image[:,:,:,e] = motion_PD(label[:,:,:,e])
    return image, label

def NiiDataset(dir, transform):
    return Split4th(NdarrayDataset(dir, transform, read = r1))

def PdDataset(dir, transform):
    return Split4th(NdarrayDataset(dir, transform, read = r2))

def NiiDatasetSim(dir, transform):
    return Split4th(NdarrayDataset(dir, transform, read = r3))