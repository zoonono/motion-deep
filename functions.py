import numpy as np
from scipy.interpolate import interp2d
import torch
from torch.autograd import Variable

def interp_double(arr):
    x = np.array(list(range(1, arr.shape[1] + 1)))
    y = np.array(list(range(1, arr.shape[0] + 1)))
    i = interp2d(x, y, arr)
    x2 = double_length(len(x))
    y2 = double_length(len(y))
    return i(x2, y2)
    
def double_length(x):
    return np.array(list(range(x * 2 - 1, x * (x * 2 - 1) + 1,  
            x - 1))) / (x * 2 - 1)
            
def double_weight(arr):
    nrr = np.zeros(np.array(arr.shape) * np.array((1,1,2,2)) 
                   + np.array((0,0,1,1)))
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            nrr[i][j] = zero_pad(interp_double(arr[i][j]))
    return nrr
    
def d_out(d_in, padding, kernel, stride):
    return 1 + int((d_in + 2 * padding - kernel)/stride)

def zero_pad(arr, n = 1):
    return np.pad(arr, ((0, n),), 'constant')
    
def compute_loss(dataset, criterion):
    avg = 0.0
    for i, example in enumerate(dataset):
        image, label = example['image'], example['label']
        image, label = Variable(image), Variable(label)
        image, label = image.unsqueeze(0), label.unsqueeze(0) # add batch dim
        output = net(image)
        avg += (criterion(output, label).data[0] - avg) / (i + 1)
    return avg