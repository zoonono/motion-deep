import torch
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from model import VNet, num_flat_features
from data import GenericFilenames, MotionCorrDataset, ToTensor, Transpose4D
from torchvision import transforms
import time

num_epochs = 1
display_every_i = 10

t = transforms.Compose([Transpose4D(), ToTensor()])
filenames = GenericFilenames('motion_data_resid/', 'motion_corrupt_',
                             'motion_resid_', '.npy', 128)
dataset = MotionCorrDataset(filenames, lambda x: np.load(x), transform = t)
dataloader = DataLoader(dataset, batch_size = 1, shuffle = True)

net = VNet()
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)

for epoch in range(num_epochs):

    running_loss = 0.0
    for i, example in enumerate(dataloader):
        image, label = example['image'], example['label']
        image, label = Variable(image), Variable(label)
        label = label.view(-1, num_flat_features(label))

        optimizer.zero_grad()

        output = net(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]
        if i % display_every_i == display_every_i - 1:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / display_every_i))
            running_loss = 0.0

print('Finished Training')
