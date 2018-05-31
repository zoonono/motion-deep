import numpy as np
import torch
from torch.autograd import Variable

from options import load_options
from functions import compute_loss, Logger, to_boolean

from os import mkdir
from os.path import exists, join
import sys
import time
            
def main():
    if len(sys.argv) not in (4, 5, 6):
        print('Arguments:')
        print('1: Name of experiment (must be defined in options.py)')
        print('2: Num epochs')
        print('3: Load existing?')
        print('4 (optional): Cuda device number')
        print('5 (optional): Babysit')
        print('Example:')
        print('python3 train.py dncnn_mag 3 False 0 False')
        return
    
    # Parse cmd line arguments
    name = sys.argv[1]
    save = join('../models', name)
    epochs = int(sys.argv[2])
    load = to_boolean(sys.argv[3])
    device = int(len(sys.argv) >= 5 and sys.argv[4])
    babysit = sys.argv >= 6 and to_boolean(sys.argv[5])
    
    # Load cmd line arguments
    options = load_options(name)
    train, test = options['dataset']
    model = options['model']
    criterion = options['criterion']
    optimizer = options['optimizer']
    
    # Prepare variables for monitoring
    test_i = len(train) // 2
    disp_i = len(train) // 10
    sys.stdout = Logger(join(save, 'log.txt'))
    losses = None
    
    # Load model if resuming
    if not exists(save):
        mkdir(save)
    elif load:
        print('Loading model:', name)
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
            map_location = None
            model.cuda()
        else: # torch.load needs to be altered if model was saved on GPU
            map_location = lambda storage, loc: storage
        model.load_state_dict(torch.load(join(save, 'model.pth'),
                                         map_location = map_location))
        losses = np.load(join(save, 'losses.npy'))
        
    # Train the model
    print('Beginning training...')
    print('Name:', name)
    print('Epochs:', epochs)
    print('Examples per epoch:', len(train))
    start = time.time()
    for e in range(epochs):
        train_loss = 0.0
        train.shuffle()
        
        for i, example in enumerate(train):
            # Load example
            image, label = example['image'], example['label']
            if torch.cuda.is_available():
                image, label = image.cuda(), label.cuda()
            # Add batch dimension (1)
            image = Variable(image).unsqueeze(0)
            label = Variable(label).unsqueeze(0)
                    
            # Do one step
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            
            # Monitor progress
            train_loss = ((loss.data[0] - train_loss) / (i % disp_i + 1))
            if i % disp_i == disp_i - 1:
                print('[%d, %d] Train loss: %.3f, Time elapsed: %.3f'
                      % (e + 1, i + 1, train_loss, time.time() - start))
                train_loss = 0.0
            if i % test_i == test_i - 1:
                test_loss = compute_loss(test, criterion, model)
                if losses is None:
                    losses = np.array([train_loss, test_loss])
                else:
                    losses = np.vstack((losses, [train_loss, test_loss]))
                print('[%d, %d] Test loss: %.3f, Time elapsed: %.3f'
                      % (e + 1, i + 1, test_loss, time.time() - start))
            if babysit:
                print('[%d, %d] Train loss: %.3f, Time elapsed: %.3f'
                      % (e + 1, i + 1, train_loss, time.time() - start))
        # Save every epoch
        torch.save(model.state_dict(), join(save, 'model.pth'))
        np.save(join(save, 'losses.npy'), losses)
    
    print('Finished %d epochs.' % (epochs))
    print('Time elapsed: %.3f' % (time.time() - start))
    
if __name__ == '__main__':
    main()
    