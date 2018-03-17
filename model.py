import torch
import torch.nn as nn
import numpy as np

from math import ceil

def pad(d_in, d_out, kernel, stride):
    """Returns padding such that a convolution produces d_out given d_in.
    Only works if such a padding is possible.
    out = (1 / stride)(in + 2 * padding - kernel) + 1
    """
    return int(ceil(((d_out - 1) * stride + kernel - d_in) / 2))

def pad_out_t(d_in, d_out, kernel, stride):
    """Pads the output of conv_transpose, d_out >= (d_in - 1) * stride + kernel
    d_out = (d_in - 1) * stride + kernel
    """
    return int(d_out - ((d_in - 1) * stride + kernel)) # np.int32 -> int

def pad_full(d_in, d_out, kernel, stride):
    return tuple([pad(d_in[i], d_out[i], kernel, stride)
                  for i in range(len(d_in))])

def pad_out_t_full(d_in, d_out, kernel, stride):
    return tuple([pad_out_t(d_in[i], d_out[i], kernel, stride)
                  for i in range(len(d_in))])

def concat(a, b):
    """Assumes a and b are B x C x H x W x D or B x C x H x W.
    """
    return torch.cat((a, b), 1)

def tile_add(x, y):
    """Assumes x and y are B x C x H x W x D or B x C x H x W. Adds x and y elementwise by tiling
    the smaller along the C dimension. H, W, and D must be the same and C_x
    must be a multiple of C_y (or the other way around).
    """
    if x.shape[1] > y.shape[1]:
        x, y = y, x
    ratio = y.shape[1] // x.shape[1]
    if len(x.shape) == 4:
        return x.repeat(1, ratio, 1, 1) + y
    return x.repeat(1, ratio, 1, 1, 1) + y

def num_flat_features(self, x):
    size = x.size()[1:] # all dimensions but the batch
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

class VNet(nn.Module):
    """V-net architecture: https://arxiv.org/abs/1606.04797
    """
    def __init__(self, size):
        super(VNet, self).__init__()
        self.size = np.array(size)

        self.conv_dict = {}
        self.downconv1 = self.conv3d_padded(16, 16, 2, 2,
                            self.size, self.size // 2, "downconv1")
        self.downconv2 = self.conv3d_padded(32, 32, 2, 2,
                            self.size // 2, self.size // 4, "downconv2")
        self.downconv3 = self.conv3d_padded(64, 64, 2, 2,
                            self.size // 4, self.size // 8, "downconv3")
        self.downconv4 = self.conv3d_padded(128, 128, 2, 2,
                            self.size // 8, self.size // 16, "downconv4")
        self.upconv5 = self.conv3d_t_padded(256, 128, 2, 2,
                            self.size // 16, self.size // 8, "upconv5")
        self.upconv6 = self.conv3d_t_padded(128, 64, 2, 2,
                            self.size // 8, self.size // 4, "upconv6")
        self.upconv7 = self.conv3d_t_padded(64, 32, 2, 2,
                            self.size // 4, self.size // 2, "upconv7")
        self.upconv8 = self.conv3d_t_padded(32, 16, 2, 2,
                            self.size // 2, self.size, "upconv8")
        self.prelu1 = nn.PReLU(num_parameters = 16)
        self.prelu2 = nn.PReLU(num_parameters = 32)
        self.prelu3 = nn.PReLU(num_parameters = 64)
        self.prelu4 = nn.PReLU(num_parameters = 128)
        self.prelu8 = nn.PReLU(num_parameters = 16)
        self.prelu7 = nn.PReLU(num_parameters = 32)
        self.prelu6 = nn.PReLU(num_parameters = 64)
        self.prelu5 = nn.PReLU(num_parameters = 128)
        self.prelu9 = nn.PReLU(num_parameters = 16)

        self.conv10 = self.conv3d_padded(16, 1, 1, 1, self.size, self.size, "conv10_1")

    def conv3d_padded(self, ch_in, ch_out, kernel, stride, d_in, d_out, name):
        if not name in self.conv_dict:
            conv = nn.Conv3d(ch_in, ch_out, kernel, stride = stride,
                padding = pad_full(d_in, d_out, kernel, stride))
            self.add_module(name, conv)
            self.conv_dict[name] = conv
        return self.conv_dict[name]

    def conv3d_t_padded(self, ch_in, ch_out, kernel, stride, d_in, d_out, name):
        if not name in self.conv_dict:
            conv = nn.ConvTranspose3d(ch_in, ch_out, kernel, stride = stride,
                output_padding = pad_out_t_full(d_in, d_out, kernel, stride))
            self.add_module(name, conv)
            self.conv_dict[name] = conv
        return self.conv_dict[name]
    
    def load_state_dict(self, state_dict):
        self.conv3d_padded(1, 16, 5, 1, self.size, self.size, "conv1_1")
        self.conv3d_padded(16, 32, 5, 1, self.size // 2, self.size // 2, "conv2_1")
        self.conv3d_padded(32, 32, 5, 1, self.size // 2, self.size // 2, "conv2_2")
        self.conv3d_padded(32, 64, 5, 1, self.size // 4, self.size // 4, "conv3_1")
        self.conv3d_padded(64, 64, 5, 1, self.size // 4, self.size // 4, "conv3_2")
        self.conv3d_padded(64, 64, 5, 1, self.size // 4, self.size // 4, "conv3_3")
        self.conv3d_padded(64, 128, 5, 1, self.size // 8, self.size // 8, "conv4_1")
        self.conv3d_padded(128, 128, 5, 1, self.size // 8, self.size // 8, "conv4_2")
        self.conv3d_padded(128, 128, 5, 1, self.size // 8, self.size // 8, "conv4_3")
        self.conv3d_padded(128, 256, 5, 1, self.size // 16, self.size // 16, "conv5_1")
        self.conv3d_padded(256, 256, 5, 1, self.size // 16, self.size // 16, "conv5_2")
        self.conv3d_padded(256, 256, 5, 1, self.size // 16, self.size // 16, "conv5_3")
        self.conv3d_padded(256, 128, 5, 1, self.size // 8, self.size // 8, "conv6_1")
        self.conv3d_padded(128, 128, 5, 1, self.size // 8, self.size // 8, "conv6_2")
        self.conv3d_padded(128, 128, 5, 1, self.size // 8, self.size // 8, "conv6_3")
        self.conv3d_padded(128, 64, 5, 1, self.size // 4, self.size // 4, "conv7_1")
        self.conv3d_padded(64, 64, 5, 1, self.size // 4, self.size // 4, "conv7_2")
        self.conv3d_padded(64, 64, 5, 1, self.size // 4, self.size // 4, "conv7_3")
        self.conv3d_padded(64, 32, 5, 1, self.size // 2, self.size // 2, "conv8_1")
        self.conv3d_padded(32, 32, 5, 1, self.size // 2, self.size // 2, "conv8_2")
        self.conv3d_padded(32, 16, 5, 1, self.size, self.size, "conv9_1")
        super(VNet, self).load_state_dict(state_dict)
    
    def conv1(self, x):
        a = x.clone()
        x = self.conv3d_padded(1, 16, 5, 1, self.size, self.size, "conv1_1")(x)
        x = tile_add(a, x)
        return x

    def conv2(self, x):
        a = x.clone()
        x = self.conv3d_padded(16, 32, 5, 1, self.size // 2, self.size // 2, "conv2_1")(x)
        x = self.conv3d_padded(32, 32, 5, 1, self.size // 2, self.size // 2, "conv2_2")(x)
        x = tile_add(a, x)
        return x

    def conv3(self, x):
        a = x.clone()
        x = self.conv3d_padded(32, 64, 5, 1, self.size // 4, self.size // 4, "conv3_1")(x)
        x = self.conv3d_padded(64, 64, 5, 1, self.size // 4, self.size // 4, "conv3_2")(x)
        x = self.conv3d_padded(64, 64, 5, 1, self.size // 4, self.size // 4, "conv3_3")(x)
        x = tile_add(a, x)
        return x

    def conv4(self, x):
        a = x.clone()
        x = self.conv3d_padded(64, 128, 5, 1, self.size // 8, self.size // 8, "conv4_1")(x)
        x = self.conv3d_padded(128, 128, 5, 1, self.size // 8, self.size // 8, "conv4_2")(x)
        x = self.conv3d_padded(128, 128, 5, 1, self.size // 8, self.size // 8, "conv4_3")(x)
        x = tile_add(a, x)
        return x

    def conv5(self, x):
        a = x.clone()
        x = self.conv3d_padded(128, 256, 5, 1, self.size // 16, self.size // 16, "conv5_1")(x)
        x = self.conv3d_padded(256, 256, 5, 1, self.size // 16, self.size // 16, "conv5_2")(x)
        x = self.conv3d_padded(256, 256, 5, 1, self.size // 16, self.size // 16, "conv5_3")(x)
        x = tile_add(a, x)
        return x

    def conv6(self, x, cat):
        a = x.clone() # 128 16 16 8
        x = concat(x, cat) # 256 16 16 8
        x = self.conv3d_padded(256, 128, 5, 1, self.size // 8, self.size // 8, "conv6_1")(x)
        x = self.conv3d_padded(128, 128, 5, 1, self.size // 8, self.size // 8, "conv6_2")(x)
        x = self.conv3d_padded(128, 128, 5, 1, self.size // 8, self.size // 8, "conv6_3")(x)
        x = tile_add(a, x) # 128 16 16 8
        return x

    def conv7(self, x, cat):
        a = x.clone()
        x = concat(x, cat)
        x = self.conv3d_padded(128, 64, 5, 1, self.size // 4, self.size // 4, "conv7_1")(x)
        x = self.conv3d_padded(64, 64, 5, 1, self.size // 4, self.size // 4, "conv7_2")(x)
        x = self.conv3d_padded(64, 64, 5, 1, self.size // 4, self.size // 4, "conv7_3")(x)
        x = tile_add(a, x)
        return x

    def conv8(self, x, cat):
        a = x.clone()
        x = concat(x, cat)
        x = self.conv3d_padded(64, 32, 5, 1, self.size // 2, self.size // 2, "conv8_1")(x)
        x = self.conv3d_padded(32, 32, 5, 1, self.size // 2, self.size // 2, "conv8_2")(x)
        x = tile_add(a, x)
        return x

    def conv9(self, x, cat):
        a = x.clone()
        x = concat(x, cat)
        x = self.conv3d_padded(32, 16, 5, 1, self.size, self.size, "conv9_1")(x)
        x = tile_add(a, x)
        return x

    def forward(self, x):
        x1 = self.conv1(x) # (1, 128, 128, 68) -> (16, 128, 128, 64)
        x = self.prelu1(self.downconv1(x1)) # (16, 128, 128, 64) -> (16, 64, 64, 32)
        x2 = self.conv2(x) # (16, 64, 64, 32) -> (32, 64, 64, 32)
        x = self.prelu2(self.downconv2(x2)) # (32, 64, 64, 32) -> (32, 32, 32, 16)
        x3 = self.conv3(x) # (32, 32, 32, 16) -> (64, 32, 32, 16)
        x = self.prelu3(self.downconv3(x3)) # (64, 32, 32, 16) -> (64, 16, 16, 8)
        x4 = self.conv4(x) # (64, 16, 16, 8) -> (128, 16, 16, 8)
        x = self.prelu4(self.downconv4(x4)) # (128, 16, 16, 8) -> (128, 8, 8, 4)
        # (128, 8, 8, 4) --conv-> (256, 8, 8, 4) --upconv-> (128, 16, 16, 8)
        x = self.prelu5(self.upconv5(self.conv5(x)))
        # --concat-> (256, 16, 16, 8)
        # (256, 16, 16, 8) -> (128, 16, 16, 8) -> (64, 32, 32, 16)
        x = self.prelu6(self.upconv6(self.conv6(x, x4)))
        # -> (128, 32, 32, 16)
        # (128, 32, 32, 16) -> (32, 64, 64, 32)
        x = self.prelu7(self.upconv7(self.conv7(x, x3)))
        # -> (64, 64, 64, 32)
        # (64, 64, 64, 32) -> (16, 128, 128, 64)
        x = self.prelu8(self.upconv8(self.conv8(x, x2)))
        # -> (32, 128, 128, 64)
        # (32, 128, 128, 64) -> (16, 128, 128, 68)
        x = self.prelu9(self.conv9(x, x1))

        x = self.conv10(x) #(16, 128, 128, 68) -> (1, 128, 128, 68)
        return x

class VNet2d(nn.Module):
    """V-net, but in 2D
    cat and tileadd should still work since C is still dim 1
    """
    def __init__(self, size):
        super(VNet2d, self).__init__()
        self.size = np.array(size)

        self.conv_dict = {}
        self.downconv1 = self.conv2d_padded(16, 16, 2, 2,
                            self.size, self.size // 2, "downconv1")
        self.downconv2 = self.conv2d_padded(32, 32, 2, 2,
                            self.size // 2, self.size // 4, "downconv2")
        self.downconv3 = self.conv2d_padded(64, 64, 2, 2,
                            self.size // 4, self.size // 8, "downconv3")
        self.downconv4 = self.conv2d_padded(128, 128, 2, 2,
                            self.size // 8, self.size // 16, "downconv4")
        self.upconv5 = self.conv2d_t_padded(256, 128, 2, 2,
                            self.size // 16, self.size // 8, "upconv5")
        self.upconv6 = self.conv2d_t_padded(128, 64, 2, 2,
                            self.size // 8, self.size // 4, "upconv6")
        self.upconv7 = self.conv2d_t_padded(64, 32, 2, 2,
                            self.size // 4, self.size // 2, "upconv7")
        self.upconv8 = self.conv2d_t_padded(32, 16, 2, 2,
                            self.size // 2, self.size, "upconv8")
        self.prelu1 = nn.PReLU(num_parameters = 16)
        self.prelu2 = nn.PReLU(num_parameters = 32)
        self.prelu3 = nn.PReLU(num_parameters = 64)
        self.prelu4 = nn.PReLU(num_parameters = 128)
        self.prelu8 = nn.PReLU(num_parameters = 16)
        self.prelu7 = nn.PReLU(num_parameters = 32)
        self.prelu6 = nn.PReLU(num_parameters = 64)
        self.prelu5 = nn.PReLU(num_parameters = 128)
        self.prelu9 = nn.PReLU(num_parameters = 16)

        self.conv10 = self.conv2d_padded(16, 1, 1, 1, self.size, self.size, "conv10_1")

    def conv2d_padded(self, ch_in, ch_out, kernel, stride, d_in, d_out, name):
        if not name in self.conv_dict:
            conv = nn.Conv2d(ch_in, ch_out, kernel, stride = stride,
                padding = pad_full(d_in, d_out, kernel, stride))
            self.add_module(name, conv)
            self.conv_dict[name] = conv
        return self.conv_dict[name]

    def conv2d_t_padded(self, ch_in, ch_out, kernel, stride, d_in, d_out, name):
        if not name in self.conv_dict:
            conv = nn.ConvTranspose2d(ch_in, ch_out, kernel, stride = stride,
                output_padding = pad_out_t_full(d_in, d_out, kernel, stride))
            self.add_module(name, conv)
            self.conv_dict[name] = conv
        return self.conv_dict[name]
    
    def load_state_dict(self, state_dict):
        self.conv2d_padded(1, 16, 5, 1, self.size, self.size, "conv1_1")
        self.conv2d_padded(16, 32, 5, 1, self.size // 2, self.size // 2, "conv2_1")
        self.conv2d_padded(32, 32, 5, 1, self.size // 2, self.size // 2, "conv2_2")
        self.conv2d_padded(32, 64, 5, 1, self.size // 4, self.size // 4, "conv3_1")
        self.conv2d_padded(64, 64, 5, 1, self.size // 4, self.size // 4, "conv3_2")
        self.conv2d_padded(64, 64, 5, 1, self.size // 4, self.size // 4, "conv3_3")
        self.conv2d_padded(64, 128, 5, 1, self.size // 8, self.size // 8, "conv4_1")
        self.conv2d_padded(128, 128, 5, 1, self.size // 8, self.size // 8, "conv4_2")
        self.conv2d_padded(128, 128, 5, 1, self.size // 8, self.size // 8, "conv4_3")
        self.conv2d_padded(128, 256, 5, 1, self.size // 16, self.size // 16, "conv5_1")
        self.conv2d_padded(256, 256, 5, 1, self.size // 16, self.size // 16, "conv5_2")
        self.conv2d_padded(256, 256, 5, 1, self.size // 16, self.size // 16, "conv5_3")
        self.conv2d_padded(256, 128, 5, 1, self.size // 8, self.size // 8, "conv6_1")
        self.conv2d_padded(128, 128, 5, 1, self.size // 8, self.size // 8, "conv6_2")
        self.conv2d_padded(128, 128, 5, 1, self.size // 8, self.size // 8, "conv6_3")
        self.conv2d_padded(128, 64, 5, 1, self.size // 4, self.size // 4, "conv7_1")
        self.conv2d_padded(64, 64, 5, 1, self.size // 4, self.size // 4, "conv7_2")
        self.conv2d_padded(64, 64, 5, 1, self.size // 4, self.size // 4, "conv7_3")
        self.conv2d_padded(64, 32, 5, 1, self.size // 2, self.size // 2, "conv8_1")
        self.conv2d_padded(32, 32, 5, 1, self.size // 2, self.size // 2, "conv8_2")
        self.conv2d_padded(32, 16, 5, 1, self.size, self.size, "conv9_1")
        super(VNet, self).load_state_dict(state_dict)
    
    def conv1(self, x):
        a = x.clone()
        x = self.conv2d_padded(1, 16, 5, 1, self.size, self.size, "conv1_1")(x)
        x = tile_add(a, x)
        return x

    def conv2(self, x):
        a = x.clone()
        x = self.conv2d_padded(16, 32, 5, 1, self.size // 2, self.size // 2, "conv2_1")(x)
        x = self.conv2d_padded(32, 32, 5, 1, self.size // 2, self.size // 2, "conv2_2")(x)
        x = tile_add(a, x)
        return x

    def conv3(self, x):
        a = x.clone()
        x = self.conv2d_padded(32, 64, 5, 1, self.size // 4, self.size // 4, "conv3_1")(x)
        x = self.conv2d_padded(64, 64, 5, 1, self.size // 4, self.size // 4, "conv3_2")(x)
        x = self.conv2d_padded(64, 64, 5, 1, self.size // 4, self.size // 4, "conv3_3")(x)
        x = tile_add(a, x)
        return x

    def conv4(self, x):
        a = x.clone()
        x = self.conv2d_padded(64, 128, 5, 1, self.size // 8, self.size // 8, "conv4_1")(x)
        x = self.conv2d_padded(128, 128, 5, 1, self.size // 8, self.size // 8, "conv4_2")(x)
        x = self.conv2d_padded(128, 128, 5, 1, self.size // 8, self.size // 8, "conv4_3")(x)
        x = tile_add(a, x)
        return x

    def conv5(self, x):
        a = x.clone()
        x = self.conv2d_padded(128, 256, 5, 1, self.size // 16, self.size // 16, "conv5_1")(x)
        x = self.conv2d_padded(256, 256, 5, 1, self.size // 16, self.size // 16, "conv5_2")(x)
        x = self.conv2d_padded(256, 256, 5, 1, self.size // 16, self.size // 16, "conv5_3")(x)
        x = tile_add(a, x)
        return x

    def conv6(self, x, cat):
        a = x.clone() # 128 16 16 8
        x = concat(x, cat) # 256 16 16 8
        x = self.conv2d_padded(256, 128, 5, 1, self.size // 8, self.size // 8, "conv6_1")(x)
        x = self.conv2d_padded(128, 128, 5, 1, self.size // 8, self.size // 8, "conv6_2")(x)
        x = self.conv2d_padded(128, 128, 5, 1, self.size // 8, self.size // 8, "conv6_3")(x)
        x = tile_add(a, x) # 128 16 16 8
        return x

    def conv7(self, x, cat):
        a = x.clone()
        x = concat(x, cat)
        x = self.conv2d_padded(128, 64, 5, 1, self.size // 4, self.size // 4, "conv7_1")(x)
        x = self.conv2d_padded(64, 64, 5, 1, self.size // 4, self.size // 4, "conv7_2")(x)
        x = self.conv2d_padded(64, 64, 5, 1, self.size // 4, self.size // 4, "conv7_3")(x)
        x = tile_add(a, x)
        return x

    def conv8(self, x, cat):
        a = x.clone()
        x = concat(x, cat)
        x = self.conv2d_padded(64, 32, 5, 1, self.size // 2, self.size // 2, "conv8_1")(x)
        x = self.conv2d_padded(32, 32, 5, 1, self.size // 2, self.size // 2, "conv8_2")(x)
        x = tile_add(a, x)
        return x

    def conv9(self, x, cat):
        a = x.clone()
        x = concat(x, cat)
        x = self.conv2d_padded(32, 16, 5, 1, self.size, self.size, "conv9_1")(x)
        x = tile_add(a, x)
        return x

    def forward(self, x):
        x1 = self.conv1(x) # (1, 128, 128, 68) -> (16, 128, 128, 64)
        x = self.prelu1(self.downconv1(x1)) # (16, 128, 128, 64) -> (16, 64, 64, 32)
        x2 = self.conv2(x) # (16, 64, 64, 32) -> (32, 64, 64, 32)
        x = self.prelu2(self.downconv2(x2)) # (32, 64, 64, 32) -> (32, 32, 32, 16)
        x3 = self.conv3(x) # (32, 32, 32, 16) -> (64, 32, 32, 16)
        x = self.prelu3(self.downconv3(x3)) # (64, 32, 32, 16) -> (64, 16, 16, 8)
        x4 = self.conv4(x) # (64, 16, 16, 8) -> (128, 16, 16, 8)
        x = self.prelu4(self.downconv4(x4)) # (128, 16, 16, 8) -> (128, 8, 8, 4)
        # (128, 8, 8, 4) --conv-> (256, 8, 8, 4) --upconv-> (128, 16, 16, 8)
        x = self.prelu5(self.upconv5(self.conv5(x)))
        # --concat-> (256, 16, 16, 8)
        # (256, 16, 16, 8) -> (128, 16, 16, 8) -> (64, 32, 32, 16)
        x = self.prelu6(self.upconv6(self.conv6(x, x4)))
        # -> (128, 32, 32, 16)
        # (128, 32, 32, 16) -> (32, 64, 64, 32)
        x = self.prelu7(self.upconv7(self.conv7(x, x3)))
        # -> (64, 64, 64, 32)
        # (64, 64, 64, 32) -> (16, 128, 128, 64)
        x = self.prelu8(self.upconv8(self.conv8(x, x2)))
        # -> (32, 128, 128, 64)
        # (32, 128, 128, 64) -> (16, 128, 128, 68)
        x = self.prelu9(self.conv9(x, x1))

        x = self.conv10(x) #(16, 128, 128, 68) -> (1, 128, 128, 68)
        return x