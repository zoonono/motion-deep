import time
import os

import torch
from torch.autograd import Variable
from torchvision import transforms
import numpy as np

from model import *
from data import *

name = 'DnCnn'
dir = 'output/'
size = (128, 128)
in_ch = 1
depth = 20
net = DnCnn(size, depth, in_ch)
net.load_state_dict(torch.load(dir + 'model' + name + '.pth'))
net.double()

a = Variable(torch.ones((256,256)))[None,None,:,:]
b = net(a)