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

def concat(a, b):
    return torch.cat((a, b), 0)

def tile_add(x, y):
    """Assumes x and y are C x H x W x D. Adds x and y elementwise by tiling
    the smaller along the C dimension. H, W, and D must be the same and C_x
    must be a multiple of C_y (or the other way around).
    """
    if x.shape[0] > y.shape[0]:
        x, y = y, x
    ratio = y.shape[0] // x.shape[0]
    return x.repeat(ratio, 1, 1, 1) + y

def num_flat_features(self, x):
    size = x.size()[1:] # all dimensions but the batch
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

class VNet(nn.Module):
    def __init__(self):
        super(VNet, self).__init__()
        # 1 x 128 x 128 x 68
        # out = (1 / stride)((in + 2 * padding) - kernel) + 1
        # V-net architecture: https://arxiv.org/abs/1606.04797
        self.conv_dict = {}
        self.downconv1 = nn.Conv3d(16, 16, 2, stride = 2)
        self.downconv2 = nn.Conv3d(32, 32, 2, stride = 2)
        self.downconv3 = nn.Conv3d(64, 64, 2, stride = 2)
        self.downconv4 = nn.Conv3d(128, 128, 2, stride = 2)
        self.upconv5 = nn.ConvTranspose3d(256, 128, 2, stride = 2)
        self.upconv6 = nn.ConvTranspose3d(128, 64, 2, stride = 2)
        self.upconv7 = nn.ConvTranspose3d(64, 32, 2, stride = 2)
        self.upconv8 = nn.ConvTranspose3d(32, 16, 2, stride = 2)

        self.fc1 = nn.Linear(16 * 128 * 128 * 68, 1 * 128 * 128 * 68)
        self.fc2 = nn.Linear(1 * 128 * 128 * 68, 1 * 128 * 128 * 68)

    def conv3d_padded(self, ch_in, ch_out, kernel, stride, d_in, d_out, name):
        if not name in conv_dict:
            conv = nn.Conv3d(ch_in, ch_out, kernel, stride = stride,
                padding = pad_full(d_in, d_out, kernel, stride))
            self.add_module(conv)
            conv_dict[name] = conv
        return conv_dict[name]

    def conv1(self, x):
        a = x.clone()
        x = self.conv3d_padded(1, 16, 5, 1, (128, 128, 68), (128, 128, 64), "conv1_1")(x)
        x = tile_add(a, x)
        return x

    def conv2(self, x):
        a = x.clone()
        x = self.conv3d_padded(16, 32, 5, 1, (64, 64, 32), (64, 64, 32), "conv2_1")(x)
        x = self.conv3d_padded(32, 32, 5, 1, (64, 64, 32), (64, 64, 32), "conv2_2")(x)
        x = tile_add(a, x)
        return x

    def conv3(self, x):
        a = x.clone()
        x = self.conv3d_padded(32, 64, 5, 1, (32, 32, 16), (32, 32, 16), "conv3_1")(x)
        x = self.conv3d_padded(64, 64, 5, 1, (32, 32, 16), (32, 32, 16), "conv3_2")(x)
        x = self.conv3d_padded(64, 64, 5, 1, (32, 32, 16), (32, 32, 16), "conv3_3")(x)
        x = tile_add(a, x)
        return x

    def conv4(self, x):
        a = x.clone()
        x = self.conv3d_padded(64, 128, 5, 1, (16, 16, 8), (16, 16, 8), "conv4_1")(x)
        x = self.conv3d_padded(128, 128, 5, 1, (16, 16, 8), (16, 16, 8), "conv4_2")(x)
        x = self.conv3d_padded(128, 128, 5, 1, (16, 16, 8), (16, 16, 8), "conv4_3")(x)
        x = tile_add(a, x)
        return x

    def conv5(self, x):
        a = x.clone()
        x = self.conv3d_padded(128, 256, 5, 1, (8, 8, 4), (8, 8, 4), "conv5_1")(x)
        x = self.conv3d_padded(256, 256, 5, 1, (8, 8, 4), (8, 8, 4), "conv5_2")(x)
        x = self.conv3d_padded(256, 256, 5, 1, (8, 8, 4), (8, 8, 4), "conv5_3")(x)
        x = tile_add(a, x)
        return x

    def conv6(self, x):
        a = x.clone()
        x = self.conv3d_padded(256, 128, 5, 1, (16, 16, 8), (16, 16, 8), "conv6_1")(x)
        x = self.conv3d_padded(128, 128, 5, 1, (16, 16, 8), (16, 16, 8), "conv6_2")(x)
        x = self.conv3d_padded(128, 128, 5, 1, (16, 16, 8), (16, 16, 8), "conv6_3")(x)
        x = tile_add(a, x)
        return x

    def conv7(self, x):
        a = x.clone()
        x = self.conv3d_padded(128, 64, 5, 1, (32, 32, 16), (32, 32, 16), "conv7_1")(x)
        x = self.conv3d_padded(64, 64, 5, 1, (32, 32, 16), (32, 32, 16), "conv7_1")(x)
        x = self.conv3d_padded(64, 64, 5, 1, (32, 32, 16), (32, 32, 16), "conv7_1")(x)
        x = tile_add(a, x)
        return x

    def conv8(self, x):
        a = x.clone()
        x = self.conv3d_padded(64, 32, 5, 1, (64, 64, 32), (64, 64, 32), "conv8_1")(x)
        x = self.conv3d_padded(32, 32, 5, 1, (64, 64, 32), (64, 64, 32), "conv8_2")(x)
        x = tile_add(a, x)
        return x

    def conv9(self, x):
        a = x.clone()
        x = self.conv3d_padded(32, 16, 5, 1, (128, 128, 64), (128, 128, 68), "conv9_1")(x)
        x = tile_add(a, x)
        return x

    def forward(self, x):
        x1 = self.conv1(x) # (1, 128, 128, 68) -> (16, 128, 128, 64)
        x = F.PReLU(self.downconv1(x1)) # (16, 128, 128, 64) -> (16, 64, 64, 32)
        x2 = self.conv2(x) # (16, 64, 64, 32) -> (32, 64, 64, 32)
        x = F.PReLU(self.downconv2(x2)) # (32, 64, 64, 32) -> (32, 32, 32, 16)
        x3 = self.conv3(x) # (32, 32, 32, 16) -> (64, 32, 32, 16)
        x = F.PReLU(self.downconv3(x3)) # (64, 32, 32, 16) -> (64, 16, 16, 8)
        x4 = self.conv4(x) # (64, 16, 16, 8) -> (128, 16, 16, 8)
        x = F.PReLU(self.downconv4(x4)) # (128, 16, 16, 8) -> (128, 8, 8, 4)
        # (128, 8, 8, 4) --conv-> (256, 8, 8, 4) --upconv-> (128, 16, 16, 8)
        # --concat-> (256, 16, 16, 8)
        x = concat(x4, F.PReLU(self.upconv5(self.conv5(x))))
        # (256, 16, 16, 8) -> (128, 16, 16, 8) -> (64, 32, 32, 16)
        # -> (128, 32, 32, 16)
        x = concat(x3, F.PReLU(self.upconv6(self.conv6(x))))
        # (128, 32, 32, 16) -> (64, 64, 64, 32)
        x = concat(x2, F.PReLU(self.upconv7(self.conv7(x))))
        # (64, 64, 64, 32) -> (32, 128, 128, 64)
        x = concat(x1, F.PReLU(self.upconv8(self.conv8(x))))
        # (32, 128, 128, 64) -> (16, 128, 128, 68)
        x = F.PReLU(self.conv9(x))

        x = x.view(-1, num_flat_features(x)) # flatten for fc
        x = F.PReLU(self.fc1(x)) # (16 * 128 * 128 * 68) -> (1 * 128 * 128 * 68)
        x = self.fc2(x) # (1 * 128 * 128 * 68) -> (1 * 12 * 128 * 68)
        return x
