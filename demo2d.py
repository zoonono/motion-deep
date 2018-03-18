import torch
from torch.autograd import Variable
import numpy as np
from model import VNet2d
from data import GenericFilenames, MotionCorrDataset, ToTensor, Transpose2d, Decimate, BatchDim
from torchvision import transforms
import time
import os

size = np.array((128, 128))

# Assumes images start as H x W x D x C
t = transforms.Compose([Transpose2d(), BatchDim(), ToTensor()])
filenames = GenericFilenames('../motion_data_resid_2d/', 'motion_corrupt_',
                             'motion_resid_', '.npy', 8704)
train_filenames, test_filenames = filenames.split((0.890625, 0.109375))
test = MotionCorrDataset(test_filenames, lambda x: np.load(x), transform = t)

criterion = torch.nn.MSELoss()

net = VNet2d(size)
save_dir = 'output/'
net.load_state_dict(torch.load(save_dir + 'model2d.pth'))

save_filenames = GenericFilenames('../motion_data_resid_2d/', 'motion_pred_loss_',
                             'motion_pred_', '.npy', 8704)
train_save_filenames, test_save_filenames = save_filenames.split((0.890625, 0.109375))

print("Generating test example predictions...")
start_time = time.time()
for i, example in enumerate(test):
    image, label = example['image'], example['label']
    image, label = Variable(image), Variable(label)
    # label = label.view(-1, num_flat_features(label))
    output = net(image)
    loss = criterion(output, label).data[0]
    print("Loss for example", i, ":", loss)
    
    loss_filename, pred_filename = test_save_filenames[i]
    np.save(pred_filename, output.data.numpy()) # need to do output.data.cpu().numpy() if cuda
    np.save(loss_filename, np.array([loss]))
print("Time elapsed:", time.time() - start_time)