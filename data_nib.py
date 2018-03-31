import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from torchvision import transforms
from scipy.signal import decimate
import time
import os

from data import *

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
        y_path = self.stem + self.numbers[i] + self.y_post + self.ext
        return x_path, y_path
        
    def __iter__(self):
        i = 0
        while i < len(self):
            yield self[i]
            i += 1

def main():
    numbers = ['0' + str(i) for i in range(60, 81) if not i in [64, 71, 72, 75, 78]]
    filenames = MotionFilenames('../motion_data_full/NC_03_Sub',
                                numbers, '_M', '', '.nii')
    load_func = lambda x: nib.load(x).get_data().__array__()
    t = transforms.Compose([RealImag(),
                            Residual()])
    dataset = MotionCorrDataset(filenames, load_func, transform = t)
    
    save_dir = '../motion_data_resid_full/'
    if not(os.path.exists(save_dir)):
        os.mkdir(save_dir)
    out_filenames = GenericFilenames(save_dir, 'motion_corrupt_',
                                     'motion_resid_', '.npy', -1)
    start_time = time.time()
    print("Saving examples...")
    i = 0
    for sample in dataset:
        x = sample['image'] # C x H x W x D x E
        y = sample['label']
        for echo in range(x.shape[4]): # save each echo as separate example
            x_file, y_file = out_filenames[i]
            i += 1
            np.save(x_file, x[:,:,:,:,echo])
            np.save(y_file, y[:,:,:,:,echo])
    print("Time elasped: " + str(time.time() - start_time))

if __name__ == '__main__':
    main()
