import time
import os

import torch
from torch.autograd import Variable
from torchvision import transforms
import numpy as np

from model import *
from data import *
from functions import *

name = 'DnCnn'
dir = 'output/'
size = (128, 128)
in_ch = 1
depth = 20
net = DnCnn(size, depth, in_ch)
net.load_state_dict(torch.load(dir + 'model' + name + '.pth'))
net.double()

t = t = transforms.Compose([ToTensor()])
filenames = GenericFilenames('../motion_data_resid_full/', 
    'motion_corrupt_', 'motion_resid_', '.npy', 128)
_, test_filenames = filenames.split((0.78125, 0.21875))
test = MotionCorrDataset(test_filenames, lambda x: np.load(x), transform = t) # C x H x W x D

criterion = torch.nn.MSELoss()

pred_filenames = GenericFilenames('../motion_data_resid_full_pred/', 
    'motion_pred_', 'motion_pred_loss_', '.npy', 128)
_, save_filenames = filenames.split((0.78125, 0.21875))


torch.cuda.set_device(1)
net.cuda()

print("Generating test example predictions...")
start_time = time.time()
for i, example in enumerate(test):
    image, label = example['image'], example['label']
    image, label = Variable(image.cuda()), Variable(label.cuda())
    image, label = image[None,:,:,:,:], label[None,:,:,:,:]
    
    output = net(image[:,0:1,:,:,:])
    loss = criterion(output, label[:,0:1,:,:,:]).data[0]
    output2 = net(image[:,1:2,:,:,:])
    loss2 = criterion(output, label[:,1:2,:,:,:]).data[0]
    print("Losses for example", i, ":", loss, loss2)
    
    loss_filename, pred_filename = test_save_filenames[i]
    # need to do output.data.cpu().numpy() if cuda
    np.save(pred_filename, np.concatenate((
        output.data.cpu().numpy()[0,:,:,:,:], 
        output2.data.cpu().numpy()[0,:,:,:,:]),
        axis = 0)) #each is B x C x H x W x D
    np.save(loss_filename, np.array([loss, loss2]))
print("Time elapsed:", time.time() - start_time)
