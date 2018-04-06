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
    
    pred, losses = None, None
    for d in range(image.shape[4]):
        output = net(image[:,0:1,:,:,d])
        loss = criterion(output, label[:,0:1,:,:,d]).data[0]
        output2 = net(image[:,1:2,:,:,d])
        loss2 = criterion(output, label[:,1:2,:,:,d]).data[0]
        
        out_stacked = np.concatenate((
                output.data.cpu().numpy()[0,:,:,:], 
                output2.data.cpu().numpy()[0,:,:,:]),
                axis = 0)[:,:,:,None] # B x C x H x W > C x H x W x D
        loss_stacked = np.array([loss, loss2])
        if pred is None:
            pred = out_stacked
            losses = loss_stacked
        else:
            pred = np.concatenate((pred, out_stacked), axis = 3)
            losses = np.vstack((losses, loss_stacked))
    print("Losses for example", i, ":", np.mean(losses, axis = 1))
    
    loss_filename, pred_filename = test_save_filenames[i]
    # need to do output.data.cpu().numpy() if cuda
    np.save(pred_filename, pred) #each is B x C x H x W x D
    np.save(loss_filename, losses)
print("Time elapsed:", time.time() - start_time)
