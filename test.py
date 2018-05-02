import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.autograd import Variable

from data import *
from transform import *
from model import *

class NdarrayPred(NdarrayDataset):
    """Loads data from NdarrayDataset and applies model.
    
    input: dataset[i] is {'image': C x H x W x D, 
                          'label': C x H x W x D}
    output: {'image': C x H x W x D, 'label': C x H x W x D,
             'pred': C x H x W x D, 'loss': 1}
    """
    def __init__(self, dir, net, criterion, transform = None):
        super().__init__(dir, transform)
        self.net = net
        self.criterion = criterion
    
    def __getitem__(self, i):
        example = super().__getitem__(i)
        x, y = example['image'], example['label']
        x = Variable(x).unsqueeze(0)
        y = Variable(y).unsqueeze(0)
        yp = self.net(x)
        loss = self.criterion(yp, y).data[0]
        return {'image': x[0].data.numpy(), 
                'label': y[0].data.numpy(), 
                'pred': yp[0].data.numpy(), 
                'loss': loss}

class NdarrayPred2d(NdarrayDataset2d):
    """Applies 2d model to split 3d data."""
    def __init__(self, dir, net, criterion, transform = None):
        super().__init__(dir, transform)
        self.net = net
        self.criterion = criterion
    
    def __getitem__(self, i):
        loss = 0.0
        image, label, pred = None, None, None
        for d in range(self.depth):
            example = super().__getitem__(i * self.depth + d)
            x, y = example['image'], example['label']
            x = Variable(x).unsqueeze(0)
            y = Variable(y).unsqueeze(0)
            yp = self.net(x)
            loss = ((self.criterion(yp, y).data[0] - loss) / (d + 1))
            
            x = x[0].data.numpy()[:,:,:,None]
            y = y[0].data.numpy()[:,:,:,None]
            yp = yp[0].data.numpy()[:,:,:,None]
            if image is None:
                image, label, pred = x, y, yp
            else:
                image = np.concatenate((image, x), axis = 3)
                label = np.concatenate((label, y), axis = 3)
                pred = np.concatenate((pred, yp), axis = 3)
        return {'image': image, 'label': label, 
                'pred': pred, 'loss': loss}

class NdarrayPredPatch(NdarrayDatasetPatch):
    def __init__(self, dir, net, criterion, transform = None, patch_R = 8):
        super().__init__(dir, transform = transform, patch_R = patch_R)
        self.net = net
        self.criterion = criterion
    
    def __getitem__(self, i):
        loss = 0.0
        image, label, pred = np.zeros((3,) + self.example['image'].shape)
        for d in range(self.depth):
            example = super().__getitem__(i * self.depth + d)
            x, y = example['image'], example['label']
            x = Variable(x).unsqueeze(0)
            y = Variable(y).unsqueeze(0)
            yp = self.net(x)
            loss = ((self.criterion(yp, y).data[0] - loss) / (d + 1))
            
            d1, d = d % self.patch_R, d // self.patch_R
            d2, d3 = d % self.patch_R, d // self.patch_R
            d1 *= self.size[0]
            d2 *= self.size[1]
            d3 *= self.size[2]
            slice = np.index_exp[:,d1:d1+self.size[0],
                                 d2:d2+self.size[1],
                                 d3:d3+self.size[2]]
                                 
            x = x[0].data.numpy()[:,:,:,:]
            y = y[0].data.numpy()[:,:,:,:]
            yp = yp[0].data.numpy()[:,:,:,:]
            image[slice] = x
            label[slice] = y
            pred[slice] = yp
            
        return {'image': image, 'label': label, 
                'pred': pred, 'loss': loss}

t = transforms.Compose([MagPhase(), PickChannel(0),
                        Residual(), ToTensor()])
name = 'dncnn_mag_256'

test = NdarrayDataset2d('../data-npy/test', transform = t)
example = test[0]['image'] # C x H x W
in_size = example.shape[1:]
in_ch = example.shape[0]

net = DnCnn(in_size, in_ch)
criterion = torch.nn.MSELoss()
net.load_state_dict(torch.load(join(name, 'model.pth'), 
                    map_location=lambda storage, loc: storage))
test = NdarrayPred2d('../data-npy/test', net, criterion, transform = t)

plt.gray()
def load(i):
    global example
    example = test[i]
def view(d):
    image = example['image'][0,:,:,d]
    label = example['label'][0,:,:,d]
    pred = example['pred'][0,:,:,d]
    plt.subplot(221)
    plt.imshow(image)
    plt.subplot(222)
    plt.imshow(label)
    plt.subplot(223)
    plt.imshow(pred)
    plt.subplot(224)
    plt.imshow(label-pred)
    print(example['loss'])
    plt.show()
        
        
