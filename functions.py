import numpy as np
from scipy.interpolate import interp2d
import torch
import torch.nn as nn
from torch.autograd import Variable

def padding(d_in, d_out, kernel, stride):
    """Finds padding such that convolution outputs d_out given d_in.
    
    Padding is one-sided.
    out = (1 / stride)(in + 2 * padding - kernel) + 1
    """
    p = lambda i, o: int((o - 1) * stride + kernel - i)
    if isinstance(d_in, (list, tuple, np.ndarray)):
        return np.array([p(i, o) for i, o in zip(d_in, d_out)])
    return p(d_in, d_out)

def padding_t(d_in, d_out, kernel, stride):
    """Finds the output padding such that conv_transpose outputs
    d_out given d_in.
    
    Output padding is one-sided.
    d_out = (d_in - 1) * stride + kernel
    """
    p = lambda i, o: int(o - ((i - 1) * stride + kernel))
    if isinstance(d_in, (list, tuple, np.ndarray)):
        return np.array([p(i, o) for i, o in zip(d_in, d_out)])
    return p(d_in, d_out)

def to_int_tuple(arr):
    l = []
    for a in arr:
        l.append(int(a))
    return tuple(l)

def conv_padded(ch_in, ch_out, kernel, stride, d_in, d_out, dim = '2d'):
    """Returns a padded conv module that takes in ch_in, d_in and
    outputs ch_out, d_out.
    """
    d_in, d_out = d_in.astype(int), d_out.astype(int)
    if dim == '2d':
        conv_f = nn.Conv2d
    elif dim == '3d':
        conv_f = nn.Conv3d
    p = padding(d_in, d_out, kernel, stride)
    conv = conv_f(ch_in, ch_out, kernel, stride = stride,
                  padding = to_int_tuple(p // 2))
    if np.all(p % 2 == 0):
        return conv
    else:
        lp = (1, 0) * int(dim[0])
        padleft = lambda x: nn.functional.pad(x, (0, 0, 0, 0) + lp)
        return MultiModule([conv, padleft])

def conv_padded_t(ch_in, ch_out, kernel, stride, d_in, d_out, dim = '2d'):
    """Returns a padded transposed conv module that takes in 
    ch_in, d_in and outputs ch_out, d_out.
    """
    d_in, d_out = d_in.astype(int), d_out.astype(int)
    if dim == '2d':
        conv_f = nn.ConvTranspose2d
    elif dim == '3d':
        conv_f = nn.ConvTranspose3d
    p = padding_t(d_in, d_out, kernel, stride)
    return conv_f(ch_in, ch_out, kernel, stride = stride,
                  output_padding = to_int_tuple(p))
        
class MultiModule(nn.Module):
    """Composes modules and one-arg functions into one module."""
    def __init__(self, module_list):
        super().__init__()
        self.module_list = module_list
        for i, module in enumerate(self.module_list):
            if isinstance(module, nn.Module):
                self.add_module("module_" + str(i), module)
    
    def forward(self, x):
        for module in self.module_list:
            x = module(x)
        return x
        
def concat(a, b):
    """Concatenates a and b on the channel dimension.
    
    Assumes a and b are B x C x ....
    """
    return torch.cat((a, b), 1)

def tile_add(x, y):
    """Adds x and y with necessary tiling in the channel dimension.
    
    Assumes x and y are B x C x H x W (x D). Adds x and y 
    elementwise by tiling the smaller along the C dimension. 
    H, W, and D must be the same and the larger channel dimension
    must be a multiple of the smaller one.
    """
    if x.shape[1] > y.shape[1]:
        x, y = y, x
    ratio = y.shape[1] // x.shape[1]
    if len(x.shape) == 4:
        return x.repeat(1, ratio, 1, 1) + y
    return x.repeat(1, ratio, 1, 1, 1) + y
    
def num_flat_features(self, x):
    """Finds the size of x flattened to batch x size."""
    size = x.size()[1:] # all dimensions but the batch
    num_features = 1
    for s in size:
        num_features *= s
    return num_features
    
def double_arr_2d(arr, kind = 'zero'):
    """Given 2d array, upsamples by 2.
    
    kind: 'zero', 'linear', or any in scipy.interpolate.interp2d
    """
    if kind == 'zero':
        arr2 = np.zeros(np.array(arr.shape) * 2)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                arr2[2*i:2*i+2, 2*j:2*j+2] = arr[i,j]
        return arr2
    else:
        def double_length(x):
            return np.array(list(range(x * 2 - 1, x * (x * 2 - 1) 
                   + 1, x - 1))) / (x * 2 - 1)
        x = np.array(list(range(1, arr.shape[1] + 1)))
        y = np.array(list(range(1, arr.shape[0] + 1)))
        i = interp2d(x, y, arr, kind = kind)
        x2 = double_length(len(x))
        y2 = double_length(len(y))
        return i(x2, y2)
            
def double_weight_2d(arr, kind = 'zero'):
    """Given a 2d CNN weight, upsamples each kernel by 2.
    
    kind: 'zero', 'linear', or any in scipy.interpolate.interp2d
    """
    nrr = np.zeros(np.array(arr.shape) * np.array((1,1,2,2)) 
                   + np.array((0,0,1,1)))
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            nrr[i][j] = double_arr(arr[i][j], kind)
    return nrr
    
def compute_loss(dataset, criterion, net):
    """Given a model and dataset, computes average loss."""
    avg = 0.0
    for i, example in enumerate(dataset):
        image, label = example['image'], example['label']
        if torch.cuda.is_available():
            image, label = image.cuda(), label.cuda()
        image, label = Variable(image), Variable(label)
        image, label = image.unsqueeze(0), label.unsqueeze(0)
        output = net(image)
        avg += (criterion(output, label).data[0] - avg) / (i + 1)
    return avg
