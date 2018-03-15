import torch
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from model import VNet
from data import GenericFilenames, MotionCorrDataset, ToTensor, Transpose4D, Decimate
from torchvision import transforms
import time

def compute_loss(dataset, criterion):
    running_loss = 0.0
    for example in dataset:
        image, label = example['image'], example['label']
        image, label = Variable(image), Variable(label)
        image, label = image[None,:,:,:,:], label[None,:,:,:,:] # add batch dim
        # label = label.view(-1, num_flat_features(label))
        output = net(image)
        running_loss += criterion(output, label).data[0]
    return running_loss / len(dataset)

num_epochs = 1
display_every_i = 10
size = np.array((128, 128, 68))

# Assumes images start as H x W x D x C
t = transforms.Compose([Transpose4D(), ToTensor()])
filenames = GenericFilenames('../motion_data_resid/', 'motion_corrupt_',
                             'motion_resid_', '.npy', 128)
train_filenames, test_filenames = filenames.split((0.78125, 0.21875))
train = MotionCorrDataset(train_filenames, lambda x: np.load(x), transform = t)
test = MotionCorrDataset(test_filenames, lambda x: np.load(x), transform = t)

net = VNet(size)
criterion = torch.nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)

losses = []
print('Beginning Training...')
for epoch in range(num_epochs):

    running_loss = 0.0
    for i, example in enumerate(train):
        start_time = time.time()

        image, label = example['image'], example['label']
        image, label = Variable(image), Variable(label)
        image, label = image[None,:,:,:,:], label[None,:,:,:,:] # add batch dim
        # label = label.view(-1, num_flat_features(label))

        optimizer.zero_grad()

        output = net(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]
        if i % display_every_i == display_every_i - 1:
            test_loss = compute_loss(test, criterion)
            train_loss = running_loss / display_every_i
            print('[%d, %5d] Training loss: %.3f, Test loss: %.3f, Time: %.3f' %
                  (epoch + 1, i + 1, train_loss, test_loss, time.time() - start_time))
            running_loss = 0.0
            losses.append([train_loss, test_loss])

torch.save(net.state_dict(), 'output/model.pth')
np.save(np.array(losses), 'output/loss.npy')
print('Finished Training')
