import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import data
from math import ceil

def pad(d_in, d_out, kernel, stride):
    """Returns padding such that a convolution produces d_out given d_in.
    Only works if such a padding is possible."""
	return ceil(((d_out - 1) * stride + kernel - d_in) / 2)

def pad_full(d_in, d_out, kernel, stride):
    return tuple([pad(d_in[i], d_out[i], kernel, stride)
                  for i in range(len(d_in))])

def conv3d_padded(ch_in, ch_out, kernel, stride, d_in, d_out):
    return nn.Conv3d(ch_in, ch_out, kernel, stride = stride,
        padding = pad_full(d_in, d_out, kernel, stride))

def tile_add(x, y):
    """Assumes x and y are C x H x W x D. Adds x and y elementwise by tiling
    the smaller along the C dimension. H, W, and D must be the same and C_x
    must be a multiple of C_y (or the other way around).
    """
    if x.shape[0] > y.shape[0]:
        x, y = y, x
    ratio = y.shape[0] // x.shape[0]
    return x.repeat(ratio, 1, 1, 1) + y

class Net(nn.Module):
"""TODO: construct nn"""
    def __init__(self):
        super(Net, self).__init__()
        # 1 x 128 x 128 x 68
        # out = (1 / stride)((in + 2 * padding) - kernel) + 1
        # V-net architecture: https://arxiv.org/abs/1606.04797
        self.conv1 = lambda x: (nn.Conv3d(16, 16, 2, stride = 2)
                                (tile_add(x, nn.Conv3d(1, 16, 5)(x))))
        self.conv2 = nn.Conv2d(16, 32, 2, stride = 2)
        self.conv2 = nn.Conv2d(32, 64, 2, stride = 2)
        self.conv2 = nn.Conv2d(64, 128, 2, stride = 2)
        self.conv2 = nn.Conv2d(128, 256, 2, stride = 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def conv1(self, x):
        a = x.clone()
        x = tile_add(x, nn.Conv3d(1, 16, 5, padding = 2)(x))
        x = nn.Conv3d(16, 16, 2, stride = 2)(x)
        return x

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) # specifiying just 1 number is also ok
        x = x.view(-1, self.num_flat_features(x)) # flatten for fc
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions but the batch
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
