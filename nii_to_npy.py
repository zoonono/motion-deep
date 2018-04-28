import nibabel as nib
import numpy as np

import os

def main():
    numbers = ['0' + str(i) for i in range(60, 81) 
        if not i in [64, 71, 72, 75, 78]]
    in_dir = '../data'
    out_dir = '../data-npy'
    if not(os.path.exists(out_dir)):
        os.mkdir(out_dir)
    
    i = 0
    for n in numbers:
        in_image = os.path.join(in_dir, 'NC_03_Sub' + str(n) 
            + '_M.nii')
        in_label = os.path.join(in_dir, 'NC_03_Sub' + str(n) 
            + '.nii')
        
        print(in_image, in_label)
        
        image = nib.load(in_image).get_data()
        label = nib.load(in_label).get_data()
        
        example = np.concatenate((image[None,:,:,:,:], 
            label[None,:,:,:,:]), axis = 0)
        
        # Split echoes
        for e in range(example.shape[4]):
            out = os.path.join(out_dir, 'motion_' + str(i) + '.npy')
            print('    ' + out)
            np.save(out, example[:,:,:,:,e])
            i += 1

if __name__ == '__main__':
    main()
            
        
