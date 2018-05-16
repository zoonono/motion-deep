import torch.nn as nn
import numpy as np

from functions import conv_padded, conv_padded_t, MultiModule, concat, double_weight_2d, tile_add

class DnCnn(nn.Module):
    """Implements the DnCNN architecture: https://arxiv.org/abs/1608.03981
    """
    def __init__(self, in_size, in_ch, depth = 20, kernel = 3, dropprob = 0.0):
        super().__init__()
        self.in_size = np.array(in_size)
        self.in_ch = in_ch
        self.depth = depth
        self.kernel = kernel
        self.dropprob = 0.0
        
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
        if self.dim == '2d':
            self.dropout = nn.Dropout2d(p = self.dropprob)
        else:
            self.dropout = nn.Dropout3d(p = self.dropprob)
            
    def forward(self, x):
        """Defines one forward pass given input x."""
        x = self.prelui(self.convi(x))
        for i in range(self.depth):
            conv = getattr(self, "conv" + str(i))
            post = getattr(self, "post" + str(i))
            x = post(x + conv(x))
        x = self.dropout(self.convf(x))
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

class UNet(nn.Module):
    """Implements the U-Net architecture: https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_size, in_ch, depth = 4, kernel = 3, dropprob = 0.0):
        super().__init__()
        self.in_size = np.array(in_size)
        self.in_ch = in_ch
        self.depth = depth
        self.kernel = kernel
        self.dropprob = dropprob
        
        if len(in_size) == 2:
            self.dim = '2d'
        elif len(in_size) == 3:
            self.dim = '3d'
        else:
            assert False, 'Input ' + str(in_size) + ' must be 2d or 3d'
        
        self.init_layers()
    
    def init_layers(self):
        def unet_module(ch, out_ch, size):
            # print(ch, size)
            conv1 = conv_padded(ch, out_ch, self.kernel, 1, size, size, 
                                dim = self.dim)
            conv2 = conv_padded(out_ch, out_ch, self.kernel, 1, size, size,
                                dim = self.dim)
            return MultiModule((conv1, nn.PReLU(num_parameters = out_ch), 
                                conv2, nn.PReLU(num_parameters = out_ch)))
        def post_module_down(ch, out_ch, size, out_size):
            conv = conv_padded(ch, out_ch, 2, 2, size, out_size, 
                               dim = self.dim)
            return MultiModule((conv, nn.PReLU(num_parameters = out_ch)))
        def post_module_up(ch, out_ch, size, out_size):
            conv = conv_padded_t(ch, out_ch, 2, 2, size, out_size, 
                                 dim = self.dim)
            return MultiModule((conv, nn.PReLU(num_parameters = out_ch)))
        
        size = self.in_size
        ch = 64
        
        conv, ch = unet_module(self.in_ch, ch, size), ch
        setattr(self, 'conv_d0', conv)
        conv, size = post_module_down(ch, ch, size, size // 2), size // 2
        setattr(self, 'conv2_d0', conv)
        for i in range(1, self.depth):
            conv, ch = unet_module(ch, ch * 2, size), ch * 2
            setattr(self, 'conv_d' + str(i), conv)
            conv, size = post_module_down(ch, ch, size, size // 2), size // 2
            setattr(self, 'conv2_d' + str(i), conv)
        self.conv_m0, ch = unet_module(ch, ch * 2, size), ch * 2
        # ch does not change due to feature forwarding
        self.conv2_m0, size = post_module_up(ch, ch // 2, size, size * 2), size * 2
        for i in range(0, self.depth):
            conv, ch = unet_module(ch, ch // 2, size), ch // 2
            setattr(self, 'conv_u' + str(i), conv)
            if i != self.depth - 1: # last layer is different
                conv, size = post_module_up(ch, ch // 2, size, size * 2), size * 2
                setattr(self, 'conv2_u' + str(i), conv)
        conv = conv_padded(ch, self.in_ch, 1, 1, size, size, dim = self.dim) 
        setattr(self, 'conv2_u' + str(self.depth - 1), conv)
        if self.dim == '2d':
            self.dropout = nn.Dropout2d(p = self.dropprob)
        else:
            self.dropout = nn.Dropout3d(p = self.dropprob)
    
    def forward(self, x):
        features = []        
        for i in range(self.depth):
            conv = getattr(self, 'conv_d' + str(i))
            post = getattr(self, 'conv2_d' + str(i))
            x = tile_add(x, conv(x))
            features.append(x)
            x = post(x)
        x = self.conv2_m0(tile_add(x, self.conv_m0(x)))
        for i in range(self.depth):
            conv = getattr(self, 'conv_u' + str(i))
            post = getattr(self, 'conv2_u' + str(i))
            x = tile_add(x, conv(concat(x, features[self.depth - i - 1])))
            x = post(x)
        x = self.dropout(x)
        return x
        
