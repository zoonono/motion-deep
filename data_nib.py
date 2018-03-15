import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from torchvision import transforms
from scipy.signal import decimate
import time

class MotionFilenames:
    def __init__(self, stem, numbers, x_post, y_post, ext):
        self.stem = stem
        self.numbers = numbers
        self.x_post = x_post
        self.y_post = y_post
        self.ext = ext

    def __len__(self):
        return (len(self.numbers))

    def __getitem__(self, i):
        x_path = self.stem + self.numbers[i] + self.x_post + self.ext
        y_path = self.stem + self.numbers[i] + self.x_post + self.ext
        return x_path, y_path

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
                    self.label, self.ext, self.size * p, offset = offset))
            offset += self.size * p
        return splitted_filenames

class MotionCorrDataset(Dataset):
    def __init__(self, filenames, load_func, transform = None):
        self.filenames = filenames
        self.load_func = load_func
        self.transform = transform

    def __len__(self):
        return (len(self.filenames))

    def __getitem__(self, i):
        x_path, y_path = self.filenames[i]
        x = self.load_func(x_path)
        y = self.load_func(y_path)
        sample = {'image': x, 'label': y}
        if self.transform:
            sample = self.transform(sample)
        return sample

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

        # swap color axis because
        # numpy image: H x W x D
        # torch image: C X H X W
        #           or C X D X H X W
        image = image[None,:,:,:].transpose((0, 3, 1, 2))
        label = label[None,:,:,:].transpose((0, 3, 1, 2))
        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(label)}

class Transpose4D(object):
    """Transpose ndarrays to C X D X H X W."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # numpy image: H x W x D X C
        # torch image: C X H X W X D
        if len(image.shape) == 3:
            image = image[:,:,:,None]
            label = label[:,:,:,None]
        image = image.transpose((3, 0, 1, 2))
        label = label.transpose((3, 0, 1, 2))
        return {'image': image,
                'label': label}

def main():
    numbers = ['0' + str(i) for i in range(60, 81) if not i in [64, 71, 72, 75, 78]]
    filenames = MotionFilenames('motion_data/NC_03_Sub',
                                numbers, '_dataM', '_data', '.nii')
    load_func = lambda x: nib.load(x).get_data().__array__()
    t = transforms.Compose([Decimate(factor = 2, axes = [0, 1, 2]),
                            Residual()])
    dataset = MotionCorrDataset(filenames, load_func, transform = t)
    # dataloader = DataLoader(dataset, batch_size = 1, shuffle = True)

    out_filenames = GenericFilenames('motion_data_resid/', 'motion_corrupt_',
                                     'motion_resid_', '.npy', -1)
    start_time = time.time()
    print("Saving examples...")
    i = 0
    for sample in dataset:
        x = sample['image']
        y = sample['label']
        for echo in range(x.shape[3]): # save each echo as separate example
            x_file, y_file = out_filenames[i]
            i += 1
            np.save(x_file, x[:,:,:,echo])
            np.save(y_file, y[:,:,:,echo])
    print("Time elasped: " + str(time.time() - start_time))

if __name__ == '__main__':
    main()