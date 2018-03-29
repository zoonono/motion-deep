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

def conv_padded(ch_in, ch_out, kernel, stride, d_in, d_out, dim = '2d'):
    if dim == '2d':
        conv_f = nn.Conv2d
    elif dim == '3d':
        conv_f = nn.Conv3d
    return conv_f(ch_in, ch_out, kernel, stride = stride,
                  padding = pad_full(d_in, d_out, kernel, stride))

def conv_t_padded(ch_in, ch_out, kernel, stride, d_in, d_out, dim = '2d'):
    if dim == '2d':
        conv_f = nn.ConvTranspose2d
    elif dim == '3d':
        conv_f = nn.ConvTranspose3d
    return conv_f(ch_in, ch_out, kernel, stride = stride,
                  output_padding = pad_out_t_full(d_in, d_out, kernel, stride))

class DnCnn(nn.Module):
    """Implements the DnCNN architecture: https://arxiv.org/abs/1608.03981
    """
    def __init__(self, size, depth):
        super(DnCnn, self).__init__()
        self.size = np.array(size)
        self.depth = depth

        self.conv1 = conv_padded(1, 64, 3, 1, size, size)
        self.prelu1 = nn.PReLU(num_parameters = 64)
        for i in range(self.depth):
            c_name = "conv" + str(i + 2)
            b_name = "batch" + str(i + 2)
            p_name = "prelu" + str(i + 2)
            setattr(self, c_name, conv_padded(64, 64, 3, 1, size, size))
            setattr(self, b_name, nn.BatchNorm2d(64))
            setattr(self, p_name, nn.PReLU(num_parameters = 64))
        self.convf = conv_padded(64, 1, 3, 1, size, size)

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        for i in range(self.depth):
            c_name = "conv" + str(i + 2)
            b_name = "batch" + str(i + 2)
            p_name = "prelu" + str(i + 2)
            conv = getattr(self, c_name)
            batch = getattr(self, b_name)
            prelu = getattr(self, p_name)
            x = prelu(batch(x + conv(x)))
        x = self.convf(x)
        return x

class VNet(nn.Module):
    """V-net architecture: https://arxiv.org/abs/1606.04797
    """
    def __init__(self, size, depth = 5, dim = '2d', in_ch = 1, start_ch = 16, verbose = False):
        """ Initializes layers for VNet.
        Minimum size = size // (2 ** depth)
        """
        super(VNet, self).__init__()
        size = np.array(size)
        self.size = size
        self.dim = dim
        self.depth = depth
        self.verbose = verbose
        
        ch = start_ch
        
        self.conv0 = self.resid_conv_down(in_ch, ch, size, 1, "conv0")
        self.downconv0 = conv_padded(ch, ch, 2, 2, size, size // 2, dim = self.dim)
        size //= 2
        self.prelu0 = nn.PReLU(num_parameters = ch)
        for i in range(1, depth): # down
            setattr(self, "dconv" + str(i), self.resid_conv_down(ch, ch * 2, size, 3, "dconv" + str(i)))
            ch *= 2
            setattr(self, "down" + str(i), conv_padded(ch, ch, 2, 2, size, size // 2, dim = self.dim))
            size //= 2
            setattr(self, "dprelu" + str(i), nn.PReLU(num_parameters = ch))
        # middle
        setattr(self, "mconv", self.resid_conv_down(ch, ch * 2, size, 3, "mconv"))
        ch *= 2
        setattr(self, "mup", conv_t_padded(ch, ch // 2, 2, 2, size, size * 2, dim = self.dim))
        ch //= 2
        size *= 2
        setattr(self, "mprelu", nn.PReLU(num_parameters = ch))
        for i in range(1, depth): # up
            setattr(self, "uconv" + str(i), self.resid_conv_up(ch, ch, size, 3, "uconv" + str(i)))
            setattr(self, "up" + str(i), conv_t_padded(ch, ch // 2, 2, 2, size, size * 2, dim = self.dim))
            ch //= 2
            size *= 2
            setattr(self, "uprelu" + str(i), nn.PReLU(num_parameters = ch))
        self.conv1 = self.resid_conv_up(ch, ch, size, 1, "conv1")
        self.prelu1 = nn.PReLU(num_parameters = ch)
        # recieves feature forwarding from conv0
        self.convf = conv_padded(ch, in_ch, 1, 1, size, size, dim = self.dim)

    def resid_conv_down(self, in_ch, out_ch, size, depth, name):
        """Adds necessary modules for one layer and returns a callable function.
        """
        if self.verbose:
            print(size, in_ch, out_ch, name)
        setattr(self, name + "_0", conv_padded(in_ch, out_ch, 5, 1, size, size, dim = self.dim))
        for i in range(1, depth):
            setattr(self, name + "_" + str(i), conv_padded(out_ch, out_ch, 5, 1, size, size, dim = self.dim))
        def conv(x):
            a = x.clone()
            for i in range(depth):
                if self.verbose:
                    print(x.shape)
                x = getattr(self, name + "_" + str(i))(x)
            return tile_add(a, x)
        return conv

    def resid_conv_up(self, in_ch, out_ch, size, depth, name):
        """Adds necessary modules for one layer with feature forwarding and returns a callable function.
        """
        if self.verbose:
            print(size, in_ch, out_ch, name)
        setattr(self, name + "_0", conv_padded(in_ch * 2, out_ch, 5, 1, size, size, dim = self.dim))
        for i in range(1, depth):
            setattr(self, name + "_" + str(i), conv_padded(out_ch, out_ch, 5, 1, size, size, dim = self.dim))
        def conv(x, feature):
            a = x.clone()
            x = concat(x, feature) # in_ch *= 2
            for i in range(depth):
                if self.verbose:
                    print(x.shape)
                x = getattr(self, name + "_" + str(i))(x)
            return tile_add(a, x)
        return conv

    def forward(self, x):
        x0 = self.conv0(x)
        x = self.prelu0(self.downconv0(x0))
        for i in range(1, self.depth):
            conv = getattr(self, "dconv" + str(i))
            down = getattr(self, "down" + str(i))
            prelu = getattr(self, "dprelu" + str(i))
            setattr(self, "x" + str(i), conv(x)) # save for forwarding
            x = prelu(down(getattr(self, "x" + str(i))))
        x = self.mprelu(self.mup(self.mconv(x)))
        for i in range(1, self.depth):
            forwarded = getattr(self, "x" + str(self.depth - i))
            conv = getattr(self, "uconv" + str(i))
            up = getattr(self, "up" + str(i))
            prelu = getattr(self, "uprelu" + str(i))
            x = prelu(up(conv(x, forwarded)))
        x = self.prelu1(self.conv1(x, x0))
        x = self.convf(x)
        return x
