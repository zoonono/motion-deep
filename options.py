import torch
from torchvision import transforms

from data import *
from transform import *

def load_options(name):
    if name == 'dncnn_smallm_twoch':
        """2d DnCnn trained on small motion with two-channel architecture"""
        t = transforms.Compose([RealImag(), Residual(), ToTensor()])
        train = NiiDataset2d('../data/8echo/train', transform = t)
        test = NiiDataset2d('../data/8echo/test', transform = t)
        criterion = torch.nn.MSELoss()
        depth = 20
    elif name == 'dncnn_smallm_mag':
        """2d DnCnn trained on small motion with only magnitude"""
        t = transforms.Compose([MagPhase(), PickChannel(0), Residual(), ToTensor()])
        train = NiiDataset2d('../data/8echo/train', transform = t)
        test = NiiDataset2d('../data/8echo/test', transform = t)
        criterion = torch.nn.MSELoss()
        depth = 20
    return {'train': train, 'test': test, 
            'criterion': criterion, 'depth': depth}