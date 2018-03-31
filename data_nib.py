import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from torchvision import transforms
from scipy.signal import decimate
import time

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
    # numbers = ['0' + str(i) for i in range(60, 81) if not i in [64, 71, 72, 75, 78]]
    # filenames = MotionFilenames('../motion_data/NC_03_Sub',
                                # numbers, '_dataM', '_data', '.nii')
    # load_func = lambda x: nib.load(x).get_data().__array__()
    # t = transforms.Compose([Decimate(factor = 2, axes = [0, 1, 2]),
                            # Residual()])
    # dataset = MotionCorrDataset(filenames, load_func, transform = t)
    # # dataloader = DataLoader(dataset, batch_size = 1, shuffle = True)

    # # dir = '../motion_data_resid/'
    # dir = '../motion_data_resid_2d/'
    # out_filenames = GenericFilenames(dir, 'motion_corrupt_',
                                     # 'motion_resid_', '.npy', -1)
    # start_time = time.time()
    # print("Saving examples...")
    # i = 0
    # for sample in dataset:
        # x = sample['image']
        # y = sample['label']
        # # for echo in range(x.shape[3]): # save each echo as separate example
            # # x_file, y_file = out_filenames[i]
            # # i += 1
            # # np.save(x_file, x[:,:,:,echo])
            # # np.save(y_file, y[:,:,:,echo])
        # for echo in range(x.shape[3]): # save each echo as separate example
            # for slice in range(x.shape[2]):
                # x_file, y_file = out_filenames[i]
                # i += 1
                # np.save(x_file, x[:,:,slice,echo])
                # np.save(y_file, y[:,:,slice,echo])
    # print("Time elasped: " + str(time.time() - start_time))

if __name__ == '__main__':
    main()
