import torch
import torch.optim as optim
from torchvision import transforms

from data import *
from model import DnCnn, UNet
from transform import *

def load_options(name):
    dropprob = 0.0
    train = lambda t: NiiDataset2d('../data/8echo/train', transform = t)
    test = lambda t: NiiDataset2d('../data/8echo/test', transform = t)
    criterion = torch.nn.MSELoss()
    if 'dncnn' in name:
        depth = 20
        model = DnCnn
    elif 'unet' in name:
        depth = 4
        model = UNet
    optimizer = lambda params: optim.Adam(params)
    
    if name == 'dncnn_smallm_twoch':
        """2d DnCnn trained on small motion with two-channel architecture"""
        t = transforms.Compose([RealImag(), Residual(), ToTensor()])
    elif name == 'dncnn_smallm_mag':
        """2d DnCnn trained on small motion with only magnitude"""
        t = transforms.Compose([MagPhase(), PickChannel(0), Residual(), ToTensor()])
    elif name == 'dncnn_smallm_real':
        """2d DnCnn trained on small motion with only real"""
        t = transforms.Compose([RealImag(), PickChannel(0), Residual(), ToTensor()])
    elif name == 'dncnn_smallm_twoch_drop':
        """2d DnCnn trained on small motion with two-channel architecture and dropout"""
        t = transforms.Compose([RealImag(), Residual(), ToTensor()])
        dropprob = 0.5
    elif name == 'dncnn_smallm_mag_8and16':
        """Includes both 8echo and 16echo data"""
        t = transforms.Compose([MagPhase(), PickChannel(0), Residual(), ToTensor()])
        train1 = NiiDataset2d('../data/8echo/train', transform = t)
        test1 = NiiDataset2d('../data/8echo/test', transform = t)
        train2 = NiiDataset2d('../data/16echo/train', transform = t)
        test2 = NiiDataset2d('../data/16echo/test', transform = t)
        train = lambda t: CombinedDataset(train1, train2)
        test = lambda t: CombinedDataset(test1, test2)
    elif name == 'unet_smallm_mag':
        """2d UNet trained on small motion with only magnitude"""
        t = transforms.Compose([MagPhase(), PickChannel(0), Residual(), ToTensor()])
        optimizer = lambda params: optim.Adamax(params)
    elif name == 'dncnn_sim_e0_mag':
        t = transforms.Compose([MagPhase(), PickChannel(0), Residual(), ToTensor()])
        train = lambda t: NiiDatasetSim2d('../data/8echo/train', echo = 0, transform = t)
        test = lambda t: NiiDatasetSim2d('../data/8echo/test', echo = 0, transform = t)
    elif name == 'dncnn_sim_e0_mag_patch':
        t = transforms.Compose([MagPhase(), PickChannel(0), Residual(), ToTensor()])
        train = lambda t: NiiDatasetSimPatch('../data/8echo/train', echo = 0, transform = t)
        test = lambda t: NiiDatasetSimPatch('../data/8echo/test', echo = 0, transform = t)
    elif name == 'dncnn_sim_mag':
        t = transforms.Compose([MagPhase(), PickChannel(0), Residual(), ToTensor()])
        train = lambda t: NiiDatasetSim2dFull('../data/8echo/train', transform = t)
        test = lambda t: NiiDatasetSim2dFull('../data/8echo/test', transform = t)
    elif name == 'dncnn_sim_mag_patch':
        t = transforms.Compose([MagPhase(), PickChannel(0), Residual(), ToTensor()])
        train = lambda t: NiiDatasetSimPatchFull('../data/8echo/train', transform = t)
        test = lambda t: NiiDatasetSimPatchFull('../data/8echo/test', transform = t)
    return {'train': train(t), 'test': test(t), 'criterion': criterion, 
            'depth': depth, 'dropprob': dropprob, 'model': model,
            'optimizer': optimizer}
