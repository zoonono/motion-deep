import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms

from data import *
from model import *
from transform import *

from os import mkdir
from os.path import exists, join
import time

def main():
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    num_epochs = 5
    load = True
    t = transforms.Compose([MagPhase(), PickChannel(0),
                            ToTensor()]) #Residual(), Decimate()
    
#    name = 'dncnn_mag_patch32'
#    name = 'dncnn_phase_patch32'
#    train = NdarrayDatasetPatch('../data-npy/train', transform = t)
#    test = NdarrayDatasetPatch('../data-npy/test', transform = t)
#    name = 'dncnn_mag_256'
#    name += '_d40'
#    name += '_l1'
#    name += '_smooth'
#    name = 'dncnn_phase_256'
#    name = 'dncnn_mag_128'
#    name = 'dncnn_real_128'
    name = 'dncnn_nores_mag_256'
    train = NdarrayDataset2d('../data-npy/train', transform = t)
    test = NdarrayDataset2d('../data-npy/test', transform = t)
    
    criterion = torch.nn.MSELoss() #(Smooth)L1Loss(), MSELoss()
    depth = 20
    #####
    example = train[0]['image'] # C x H x W
    in_size = example.shape[1:]
    in_ch = example.shape[0]
    test_every_i = len(train) // 2
    display_every_i = len(train) // 10
    
    net = DnCnn(in_size, in_ch, depth = depth)
    optimizer = optim.Adam(net.parameters())
    
    losses = None
    if not exists(name):
        mkdir(name)
    if load:
        print("Loading model: " + name)
        net.load_state_dict(torch.load(join(name, 'model.pth')))
        losses = np.load(join(name, 'losses.npy'))
    
    if torch.cuda.is_available():
        net.cuda()
    
    print('Beginning Training...')
    print('Epochs:', num_epochs)
    print('Examples per epoch:', len(train))
    total_time = time.time()
    for epoch in range(num_epochs):
        train_loss, test_loss = 0.0, 0.0
        train.shuffle()
        
        for i, example in enumerate(train):        
            image, label = example['image'], example['label']
            if torch.cuda.is_available():
                image, label = image.cuda(), label.cuda()
            image = Variable(image).unsqueeze(0)
            label = Variable(label).unsqueeze(0)
                    
            optimizer.zero_grad()
            output = net(image)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            
            train_loss += ((loss.data[0] - train_loss)
                           / (i % test_every_i + 1))
            if i % test_every_i == test_every_i - 1:
                test_loss = compute_loss(test, criterion, net)
                if losses is None:
                    losses = np.array([train_loss, test_loss])
                else:
                    losses = np.vstack((losses, 
                        [train_loss, test_loss]))
                print(('[%d, %5d] Training loss: %.3f, ' 
                    % (epoch + 1, i + 1, train_loss))
                    + ('Test loss: %.3f, Time: %.3f'
                    % (test_loss, time.time() - total_time)))
                train_loss = 0.0
            if i % display_every_i == 0:
                print('[%d] Training loss: %.3f, Time: %3f' 
                    % (i + 1, train_loss, time.time() - total_time))
        torch.save(net.state_dict(), join(name, 'model.pth'))
        np.save(join(name, 'losses.npy'), losses)
    print('Time elapsed: %.3f' % (time.time() - total_time))

if __name__ == '__main__':
    main()
