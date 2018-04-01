import os
import numpy as np
from data import *
from torchvision import transforms

filenames = GenericFilenames('../motion_data_resid_2d/', 'motion_corrupt_',
                             'motion_resid_', '.npy', 8704)
train_filenames, test_filenames = filenames.split((0.890625, 0.109375))
test = MotionCorrDataset(test_filenames, lambda x: np.load(x), transform = BackDim(both = True))

save_filenames = GenericFilenames('../dncnn/', 'motion_pred_dn_',
                         'motion_pred_loss_dn_', '.npy', 8704)
train_save_filenames, test_save_filenames = save_filenames.split((0.890625, 0.109375))
t = transforms.Compose([RemoveDims(), BackDim()])
pred = MotionCorrDataset(test_save_filenames, lambda x: np.load(x), transform = t)

offset = 7752

save_dir = '../dncnn_3d/'
if not(os.path.exists(save_dir)):
    os.mkdir(save_dir)

def stack(start, end):
    name_1 = 'motion_corrupt_'
    name_2 = 'motion_resid_'
    name_3 = 'motion_pred_dn_'
    name_4 = 'motion_pred_loss_dn_'

    # H x W x D -> H x W x D (stacked on D)
    i = start
    name_1 += str(i + offset)
    name_2 += str(i + offset)
    name_3 += str(i + offset)
    name_4 += str(i + offset)
    t, p = test[i], pred[i]
    d1, d2, d3, d4 = t['image'], t['label'], p['image'], p['label']
    i += 1
    while i < end:
        t, p = test[i], pred[i]
        d1 = np.concatenate((d1, t['image']), axis = 2)
        d2 = np.concatenate((d2, t['label']), axis = 2)
        d3 = np.concatenate((d3, p['image']), axis = 2)
        d4 = np.concatenate((d4, p['label']), axis = 0)
        i += 1
    name_1 += 'to' + str(i + offset) + '.npy'
    name_2 += 'to' + str(i + offset) + '.npy'
    name_3 += 'to' + str(i + offset) + '.npy'
    name_4 += 'to' + str(i + offset) + '.npy'
    np.save(save_dir + name_1, d1)
    np.save(save_dir + name_2, d2)
    np.save(save_dir + name_3, d3)
    np.save(save_dir + name_4, d4)

# 0 to 68
# 69 to 136
# 137 to 204
# 205 to 272
# 273 to 304
# 305 to 408
stack(0, 69)
stack(69, 137)
stack(137, 205)
stack(205, 273)
stack(273, 305)
stack(305, 409)