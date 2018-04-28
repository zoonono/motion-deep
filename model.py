import torch
import torch.nn as nn
import numpy as np

from math import ceil

from functions import *

class DnCnn(nn.Module):
    """Implements the DnCNN architecture: https://arxiv.org/abs/1608.03981
    """
    def __init__(self, in_size, in_ch, depth = 20, kernel = 3):
        super().__init__()
        self.in_size = np.array(in_size)
        self.in_ch = in_ch
        self.depth = depth
        self.kernel = kernel
        
        if len(in_size) == 2:
            self.dim = '2d'
        elif len(in_size) == 3:
            self.dim = '3d'
        else:
            assert False, 'Input ' + str(in_size) + ' must be 2d or 3d'
        
        self.init_layers()
        
    def init_layers(self):
        """Initializes every layer of the CNN."""
        self.convi = conv_padded(self.in_ch, 64, self.kernel, 1,
            self.in_size, self.in_size, dim = self.dim)
        self.prelui = nn.PReLU(num_parameters = 64)
        for i in range(self.depth):
            self.add_module("conv" + str(i), conv_padded(64, 64, 
                self.kernel, 1, self.in_size, self.in_size, 
                dim = self.dim))
            self.add_module("post" + str(i), MultiModule(
                [nn.BatchNorm2d(64),
                 nn.PReLU(num_parameters = 64)]))
        self.convf = conv_padded(64, self.in_ch, self.kernel, 1,
            self.in_size, self.in_size, dim = self.dim)
            
    def forward(self, x):
        """Defines one forward pass given input x."""
        x = self.prelui(self.convi(x))
        for i in range(self.depth):
            conv = getattr(self, "conv" + str(i))
            post = getattr(self, "post" + str(i))
            x = post(x + conv(x))
        x = self.convf(x)
        return x
    
    def double(self):
        """Upsamples weights by 2 to double input size.
        Default interp is zero-order hold.
        """
        self.in_size *= 2
        self.kernel *= 2
        
        self.init_layers()
        dict = self.state_dict()
        keys = dict.keys()
        for key in keys:
            if key.startswith('conv') and key.endswith('weight'):
                dict[key] = double_weight_2d(dict[key])

class VNet(nn.Module):
    """Implements the V-Net architecture: https://arxiv.org/abs/1606.04797
    """
    def __init__(self, in_size, in_ch, depth = 5, kernel = 2):
        super().__init__()
        self.in_size = np.array(in_size)
        self.in_ch = in_ch
        self.depth = 5
        
        size = self.in_size
        ch = 16
        
        
