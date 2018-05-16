import numpy as np
import torch
from torch.autograd import Variable

from options import load_options
from functions import compute_loss

from os import mkdir
from os.path import exists, join
import sys
import time

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a+")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

def main():
    if len(sys.argv) not in (4, 5, 6):
        print('Arguments:')
        print('1: Name of experiment (must be defined in options.py)')
        print('2: Num epochs')
        print('3: Load existing?')
        print('4 (optional): Cuda device number')
        print('5 (optional): Babysit')
        print('Example:')
        print('python3 train.py dncnn_smallm_mag 3 False 0 False')
        return
    name = sys.argv[1]
    num_epochs = int(sys.argv[2])
    load = sys.argv[3].lower() in ('true', 'yes', 't', '1')
    if torch.cuda.is_available():
        if len(sys.argv) >= 5:
            torch.cuda.set_device(int(sys.argv[4]))
        else:
            torch.cuda.set_device(0)
    if len(sys.argv) == 6:
        babysit = sys.argv[5].lower() in ('true', 'yes', 't', '1')
    else:
        babysit = False
    
    options = load_options(name)
    train = options['train']
    test = options['test']
    criterion = options['criterion']
    depth = options['depth']
    dropprob = options['dropprob']
    model = options['model']
    optimizer = options['optimizer']
    
    example = train[0]['image'] # C x H x W
    in_size = example.shape[1:]
    in_ch = example.shape[0]
    test_every_i = len(train) // 2
    display_every_i = len(train) // 10
    
    net = model(in_size, in_ch, depth = depth, dropprob = dropprob)
    optimizer = optimizer(net.parameters())
    
    losses = None
    if not exists(name):
        mkdir(name)
    if load:
        print("Loading model: " + name)
        if torch.cuda.is_available():
            net.load_state_dict(torch.load(join(name, 'model.pth')))
        else:
            net.load_state_dict(torch.load(join(name, 'model.pth'), 
                    map_location=lambda storage, loc: storage))
        losses = np.load(join(name, 'losses.npy'))
    
    if torch.cuda.is_available():
        net.cuda()
    
    sys.stdout = Logger(join(name, "log.txt"))
    print('Beginning Training...')
    print('Name:', name)
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
                print('[%d] Training loss: %.3f, Time: %.3f' 
                    % (i + 1, train_loss, time.time() - total_time))
            elif babysit:
                print(train_loss)
        torch.save(net.state_dict(), join(name, 'model.pth'))
        np.save(join(name, 'losses.npy'), losses)
    print('Time elapsed: %.3f' % (time.time() - total_time))

if __name__ == '__main__':
    main()
