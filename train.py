import torch
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from model import VNet, DnCnn
from data import *
from torchvision import transforms
import time
import os

exp_name = 'dncnn1'
num_epochs = 3
# test_every_i = 60 # epoch size is 128, test 2 times per epoch
test_every_i = 5000 # epoch size is 15504
display_every_i = 500
batch_size = 1
size = np.array((256, 256)) # originally (256, 256, 136)
in_ch = 2 # 2 channels: real, imag
t = transforms.Compose([ToTensor()]) # PickChannel(0), Decimate(axes = (1, 2, 3)), Patcher((32,32,32))

#net = VNet(size, in_ch = in_ch)
net = DnCnn(size, 20, in_ch = in_ch)
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(net.parameters())

filenames = GenericFilenames('../motion_data_resid_full/', 'motion_corrupt_',
                             'motion_resid_', '.npy', 128) # Assumes images start as C x H x W x D
train_filenames, test_filenames = filenames.split((0.890625, 0.109375))
train = MotionCorrDataset(train_filenames, lambda x: np.load(x), transform = t)
test = MotionCorrDataset(test_filenames, lambda x: np.load(x), transform = t)

train, test = Splitter(train, depth = 136), Splitter(test, depth = 136)

def compute_loss(dataset, criterion):
    avg = 0.0
    for i, example in enumerate(dataset):
        image, label = example['image'], example['label']
        image, label = Variable(image), Variable(label)
        image, label = image.unsqueeze(0), label.unsqueeze(0) # add batch dim
        output = net(image)
        avg += (criterion(output, label).data[0] - avg) / (i + 1)
    return avg

losses = []
train_loss, test_loss = 0.0, 0.0
save_dir = 'output/'
if not(os.path.exists(save_dir)):
    os.mkdir(save_dir)
    
print('Beginning Training...')
total_start_time = time.time()
for epoch in range(num_epochs):

    train_loss = 0.0
    train.shuffle()
    
    image_batch, label_batch = None, None
    for i, example in enumerate(train):
        start_time = time.time()

        image, label = example['image'], example['label']
        image, label = Variable(image), Variable(label)
        image, label = image.unsqueeze(0), label.unsqueeze(0) # add batch dim
        
        if image_batch is None:
            image_batch, label_batch = image, label
        else:
            image_batch = torch.cat((image_batch, image), 0)
            label_batch = torch.cat((label_batch, label), 0)
        if i % batch_size == batch_size - 1:
            optimizer.zero_grad()

            output = net(image_batch)
            loss = criterion(output, label_batch)     
            loss.backward()
            optimizer.step()
            
            image_batch, label_batch = None, None
            
            # train_loss is a moving avg, so it lags behind test_loss
            train_loss += (loss.data[0] - train_loss) / (i % test_every_i + 1)
            if i % test_every_i == test_every_i - 1:
                test_loss = compute_loss(test, criterion)
                print('[%d, %5d] Training loss: %.3f, Test loss: %.3f, Time: %.3f' %
                      (epoch + 1, i + 1, train_loss, test_loss, time.time() - start_time))
                losses.append([train_loss, test_loss])
                train_loss = 0.0
            elif i % display_every_i == display_every_i - 1:
                print(train_loss, time.time() - start_time)
                losses.append([train_loss, test_loss])
                train_loss = 0.0
            elif i == 0:
                print(train_loss, time.time() - start_time)
    torch.save(net.state_dict(), save_dir + 'model_' + exp_name + '.pth')
    np.save(save_dir + 'loss_' + exp_name + '.npy', np.array(losses))
print('Finished Training; Time Elapsed:',  time.time() - total_start_time)
