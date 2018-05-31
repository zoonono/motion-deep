import torch
import torch.optim as optim

from data import NiiDataset, PdDataset, Split2d, SplitPatch
from functions import MultiModule
from model import DnCnn, UNet
import transform as t

def load_PD_dataset():
    tr = t.Transforms((t.MagPhase(), t.PickChannel(0), 
                       t.Resize((1, 256, 256, 60, 8))), apply_to = 'image')
    tr = MultiModule((tr, t.ToTensor()))
    test = Split2d(PdDataset('../data/PD', transform = tr))
    return test

def load_options(name, testing = False):
    """Saves experiment options under names to load in train and test"""
    if name == 'dncnn_mag':
        transform = t.Transforms((t.MagPhase(), t.PickChannel(0)), 
                                 apply_to = 'both')
        transform = MultiModule(transform, t.Residual(), t.ToTensor())
        train = Split2d(NiiDataset('../data/8echo/train', transform))
        test = Split2d(NiiDataset('../data/8echo/test', transform))
        model, depth, dropprob = DnCnn, 20, 0.0
        optimizer = optim.Adam
        criterion = torch.nn.MSELoss()
    if name == 'dncnn_mag_patch':
        transform = t.Transforms((t.MagPhase(), t.PickChannel(0)), 
                                 apply_to = 'both')
        transform = MultiModule(transform, t.Residual(), t.ToTensor())
        train = SplitPatch(NiiDataset('../data/8echo/train', transform))
        test = SplitPatch(NiiDataset('../data/8echo/test', transform))
        model, depth, dropprob = DnCnn, 20, 0.0
        optimizer = optim.Adam
        criterion = torch.nn.MSELoss()
    elif name == 'unet_mag':
        transform = t.Transforms((t.MagPhase(), t.PickChannel(0)), 
                                 apply_to = 'both')
        transform = MultiModule(transform, t.Residual(), t.ToTensor())
        train = Split2d(NiiDataset('../data/8echo/train', transform))
        test = Split2d(NiiDataset('../data/8echo/test', transform))
        model, depth, dropprob = UNet, 4, 0.0
        optimizer = optim.Adamax
        criterion = torch.nn.MSELoss()
    
    if testing: # No dropout during testing
        dropprob = 0.0
    example = train[0]['image']
    in_size = example.shape[1:]
    in_ch = example.shape[0]
    model = model(in_size, in_ch, depth = depth, dropprob = dropprob)
    optimizer = optimizer(model.parameters())
    return {'dataset': (train, test), 'model': model, 'optimizer': optimizer,
            'criterion': criterion}