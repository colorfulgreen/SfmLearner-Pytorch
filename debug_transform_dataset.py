import custom_transforms
from path import Path
from numpy import array
from imageio import imread
import numpy as np

normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                            std=[0.5, 0.5, 0.5])
train_transform = custom_transforms.Compose([
        custom_transforms.RandomHorizontalFlip(),
        custom_transforms.RandomScaleCrop(),
        custom_transforms.ArrayToTensor(),
        normalize
    ])

tgt = Path('formatted_data/2011_09_26_drive_0087_sync_03/0000000006.jpg')
refs = [Path('formatted_data/2011_09_26_drive_0087_sync_03/0000000003.jpg'),
        Path('formatted_data/2011_09_26_drive_0087_sync_03/0000000009.jpg')]
tgt_img = imread(tgt).astype(np.float32)
ref_imgs = [imread(i).astype(np.float32) for i in refs]

intrinsics = array([[241.67447,   0.     , 204.16801],
                              [  0.     , 246.28487,  59.00083],
                              [  0.     ,   0.     ,   1.     ]]).astype('float32')

train_transform([tgt_img]+ref_imgs, intrinsics)

