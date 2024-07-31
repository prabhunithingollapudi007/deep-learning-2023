from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv
import os

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]

class ChallengeDataset(Dataset):
    
    _dataframe = None
    _mode = None
    _transform = None

    def __init__(self, data, mode=None):
        self._dataframe = data
        self._mode = mode

        # if mode is train, then we will apply some data augmentation
        if mode == 'train':
            # add rotation, horizontal and vertical flip
            self._transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.RandomVerticalFlip(),
                tv.transforms.RandomRotation(30),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=train_mean, std=train_std)
            ])
        else:
            self._transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=train_mean, std=train_std)
            ])

    def __len__(self):
        # override and return the length of the dataframe
        return len(self._dataframe)
    
    def img_enhancement(self, img):
        # equalize
        img = img.astype('uint8')
        clahe = cv2.createCLAHE(tileGridSize=(8, 8))
        img = clahe.apply(img)
        # img = np.expand_dims(img, 2)
        return img

    def __getitem__(self, idx):
        # override and return a tuple of (image, label)
        # each entry in data has img file, crack_one_zero, inactive_one_zero
        item = self._dataframe.iloc[idx]
        img_path = item.iloc[0]
        img_path = Path("src_to_implement/" + img_path)
        img = imread(img_path)
        img = gray2rgb(img)
        # img = self.img_enhancement(img)
        label = [item.iloc[1], item.iloc[2]]
        label_tensor = torch.tensor(label)
        # return the image and the label as a tuple
        return (self._transform(img), 
                label_tensor)
