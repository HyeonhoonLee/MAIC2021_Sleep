#!/usr/bin/env python
# -*- coding: utf-8 -*-

## SY just for testin + HH final revised 
from glob import glob
from sklearn.model_selection import GroupKFold, StratifiedKFold
import cv2
from skimage import io
from datetime import datetime

import numpy as np
import scipy as sc
import pandas as pd
import os
import time
import math
import matplotlib.image as mpimg

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate 

import random
from multiprocessing import cpu_count
from PIL import Image
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)
from albumentations.pytorch import ToTensorV2
from scipy.ndimage.interpolation import zoom
from scipy.fft import fft


package_paths = [
      '/FMix', 
    '/packages/pytorch-image-models-master', '/../packages/pytorch-image-models-master'
]
import sys;

for pth in package_paths:
    sys.path.append(pth)
from FMix.fmix import sample_mask, make_low_freq_image, binarise_mask

CFG = {
    'fold_num': 5,
    'seed': 72,
    'model_arch': 'vit_base_patch16_224',
    'img_size': 224,
    'epochs': 10,
    'train_bs': 32,
    'valid_bs': 32,
    'T_0': 10,
    'lr': 1e-4,
    'min_lr': 1e-6,
    'weight_decay':1e-6,
    'num_workers': 4,
    'accum_iter': 2, # suppoprt to do batch accumulation for backprop with effectively larger batch size
    'verbose_step': 1,
    'device': 'cuda:0',
    'img_h': 224,
    'img_w': 224
}

def str2int(lb):
    if lb == "Wake": lb = 0
    elif lb == "N1": lb = 1
    elif lb == "N2": lb = 2
    elif lb == "N3": lb = 3
    elif lb == "REM": lb = 4
    return lb

subject_list = pd.read_csv("/DATA/trainset-for_user.csv", names = ['ID', 'IDnum', 'label'])
subject_list['label'] = subject_list['label'].apply(str2int)
print(subject_list.label.value_counts())

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True 

seed_everything(seed=CFG['seed'])

def get_img(path):
    im_bgr = cv2.imread(path)
    im_rgb = im_bgr[:, :, ::-1]
    #im_gr = cv2.imread(path, cv2.IMREAD_GRAYSCALE) # (270, 480)
    #im_gr = np.expand_dims(im_gr, axis=-1)  # (270, 480, 1)`
    #print(im_rgb)
    #return im_gr
    return im_rgb

def rand_bbox(size, lam):
    W = size[1]
    H = size[0]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

class PSMDataset(Dataset):
    def __init__(self, df, data_root, 
                 transforms=None, 
                 output_label=True, 
                 one_hot_label=False,
                 do_fmix=False, 
                 fmix_params={
                     'alpha': 1., 
                     'decay_power': 3., 
                     'shape': (CFG['img_h'], CFG['img_w']),
                     'max_soft': True, 
                     'reformulate': False
                 },
                 do_cutmix=False,
                 cutmix_params={
                     'alpha': 1,
                 }
                ):
        
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.data_root = data_root
        self.do_fmix = do_fmix
        self.fmix_params = fmix_params
        self.do_cutmix = do_cutmix
        self.cutmix_params = cutmix_params
        
        self.output_label = output_label
        self.one_hot_label = one_hot_label
        
        if output_label == True:
            self.labels = self.df['label'].values
            #print(self.labels)
            
            if one_hot_label is True:
                self.labels = np.eye(self.df['label'].max()+1)[self.labels]
                #print(self.labels)
            
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index: int):

        # get labels
        if self.output_label:
            target = self.labels[index]
          
        img  = get_img("{}/{}/{}".format(self.data_root, self.df.loc[index]['ID'], self.df.loc[index]['IDnum']))
        
        '''
        img = zoom(img, (1.0, 2.0)) # 270 * 960  ## For fft of images
        #index = np.random.randint(960-540+1) # Use all; no random
        ## Make 272 for ViT since 272=16*17
        img1 = fft(img[:, :544])
        img2 = fft(img[:, -544:])
        img1_r = np.concatenate((np.zeros((1, 272)), img1.real[:, :272], np.zeros((1, 272))), axis=0) #Each 272,272 by padding
        img1_i = np.concatenate((np.zeros((1, 272)), img1.imag[:, :272], np.zeros((1, 272))), axis=0)
        img2_r = np.concatenate((np.zeros((1, 272)), img2.real[:, :272], np.zeros((1, 272))), axis=0)
        img2_i = np.concatenate((np.zeros((1, 272)), img2.imag[:, :272], np.zeros((1, 272))), axis=0)
        img = np.concatenate((np.expand_dims(img1_r, axis = 2),
                              np.expand_dims(img1_i, axis = 2), 
                              np.expand_dims(img2_r, axis = 2),
                              np.expand_dims(img2_i, axis = 2)),
                             axis=2) # 272*272*4
        
        ## other way..
        #img = np.array([img.real[:, :270], img.imag[:, :270]])
        #img = img.transpose(1,2,0)
        '''

        if self.transforms:
            img = self.transforms(image=img)['image']
        
        if self.do_fmix and np.random.uniform(0., 1., size=1)[0] > 0.5:
            with torch.no_grad():
                #lam, mask = sample_mask(**self.fmix_params)
                
                lam = np.clip(np.random.beta(self.fmix_params['alpha'], self.fmix_params['alpha']),0.6,0.7)
                
                # Make mask, get mean / std
                mask = make_low_freq_image(self.fmix_params['decay_power'], self.fmix_params['shape'])
                mask = binarise_mask(mask, lam, self.fmix_params['shape'], self.fmix_params['max_soft'])
    
                fmix_ix = np.random.choice(self.df.index, size=1)[0]
                #fmix_img  = get_img("{}/{}".format(self.data_root, self.df.iloc[fmix_ix]['image_id']))
                fmix_img  = get_img("{}/{}/{}".format(self.data_root, self.df.loc[fmix_ix]['ID'], self.df.loc[fmix_ix]['IDnum']))


                if self.transforms:
                    fmix_img = self.transforms(image=fmix_img)['image']

                mask_torch = torch.from_numpy(mask)
                
                # mix image
                img = mask_torch*img+(1.-mask_torch)*fmix_img

                #print(mask.shape)

                #assert self.output_label==True and self.one_hot_label==True

                # mix target
                rate = mask.sum()/CFG['img_h']/CFG['img_w']
                target = rate*target + (1.-rate)*self.labels[fmix_ix]
                #print(target, mask, img)
                #assert False
        
        if self.do_cutmix and np.random.uniform(0., 1., size=1)[0] > 0.5:
            #print(img.sum(), img.shape)
            with torch.no_grad():
                cmix_ix = np.random.choice(self.df.index, size=1)[0]
                #cmix_img  = get_img("{}/{}".format(self.data_root, self.df.iloc[cmix_ix]['image_id']))
                cmix_img  = get_img("{}/{}/{}".format(self.data_root, self.df.loc[cmix_ix]['ID'], self.df.loc[cmix_ix]['IDnum']))

                if self.transforms:
                    cmix_img = self.transforms(image=cmix_img)['image']
                    
                lam = np.clip(np.random.beta(self.cutmix_params['alpha'], self.cutmix_params['alpha']),0.3,0.4)
                bbx1, bby1, bbx2, bby2 = rand_bbox((CFG['img_h'], CFG['img_w']), lam)

                img[:, bbx1:bbx2, bby1:bby2] = cmix_img[:, bbx1:bbx2, bby1:bby2]

                rate = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (CFG['img_h'] * CFG['img_w']))
                target = rate*target + (1.-rate)*self.labels[cmix_ix]
                
            #print('-', img.sum())
            #print(target)
            #assert False
                            
        # do label smoothing
        #target = torch.from_numpy(target)
        #print(type(img), type(target))
        if self.output_label == True:
            #img = np.asarray(img)
            #target  = np.asarray(target)
            return img, target
        else:
            return img

def custom(img):
    fmix = scipy.fft(img)
    return img

def get_train_transforms():
    return Compose([
            #RandomResizedCrop(CFG['img_size'], CFG['img_size']),
            #Transpose(p=0.5),
            #HorizontalFlip(p=0.5),
            #VerticalFlip(p=0.5),
            #ShiftScaleRotate(p=0.5),
            #HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            #RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            Resize(CFG['img_h'], CFG['img_w']),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            #Normalize(mean=(0.5,), std=(0.5,)),
            CoarseDropout(p=0.5),
            Cutout(p=0.5),
            ToTensorV2(p=1.0),
        ], p=1.)
  
        
def get_valid_transforms():
    return Compose([
            #CenterCrop(CFG['img_size'], CFG['img_size'], p=1.),
            Resize(CFG['img_h'], CFG['img_w']),
            #Normalize(mean=(0.5,), std=(0.5,)),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)

def get_inference_transforms():
    return Compose([
            #CenterCrop(CFG['img_size'], CFG['img_size'], p=1.),
            Resize(CFG['img_h'], CFG['img_w']),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            #Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(p=1.0),
        ], p=1.)
##def img_file(ID, IDnum):
   ## return "/DATA/%s/%s"%(ID, IDnum) #Bring with mpimg.imread(), IDnum includes *.png

'''
class PSMTrain(Dataset):    # For segmentation and zoom for images.. 
    def __init__(self, df, data_root, transformes=None, output_label=True, one_hot_trial = 1):
        self.subject_list = subject_list
        self.subject_num = len(subject_list)
        self.trial = trial

    def __len__(self):
        return self.subject_num * self.trial

    def __getitem__(self, idx):
        ID = subject_list['ID'][idx%self.subject_num]
        IDnum = subject_list['IDnum'][idx%self.subject_num]
        label = subject_list['label'][idx%self.subject_num]

        if label == 'Wake': label = np.eye(5)[0, :]
        elif label == 'N1': label = np.eye(5)[1, :]
        elif label == 'N2': label = np.eye(5)[2, :]
        elif label == 'N3': label = np.eye(5)[3, :]
        elif label == 'REM': label = np.eye(5)[4, :]

        input_img = np.mean(mpimg.imread(img_file(ID, IDnum)), axis = 2)
        zoom_img = sc.ndimage.zoom(input_img, (4.0, 1.0))
        u = 43

        accel = zoom_img[:u*3, :]
        EEG = zoom_img[u*3:u*7, :]
        EOG = zoom_img[u*7:u*9, :]
        ChinEMG = zoom_img[u*9:u*10, :]
        ECG = zoom_img[u*10:u*11, :]
        Flow = zoom_img[u*11:u*14, :]
        Therm = zoom_img[u*14:u*15, :]
        Thorax = zoom_img[u*15:u*16, :]
        Abdomen =  zoom_img[u*16:u*17, :]
        Snore = zoom_img[u*17:u*18, :]
        Audio = zoom_img[u*18:u*19, :]
        LegEMG = zoom_img[u*19:u*21, :]
        Oxy85 = zoom_img[u*21:u*23, :]
        Oxy40 = zoom_img[u*23:u*25, :]
        
        sample = {'accel':accel, 'EEG':EEG, 'EOG':EOG, 'ChinEMG':ChinEMG, 'ECG':ECG, 
                  'Flow':Flow, 'Therm':Therm, 'Thorax':Thorax, 'Abdomen':Abdomen, 
                  'Snore':Snore, 'Audio':Audio, 'LegEMG':LegEMG, 'Oxy85':Oxy85, 'Oxy40':Oxy40,
                  'images':zoom_image,
                  'label':label}
        return sample
'''

def prepare_dataloader(df, trn_idx, val_idx, data_root='/DATA'):

    #from catalyst.data.sampler import BalanceClassSampler

    train_ = df.loc[trn_idx,:].reset_index(drop=True)
    valid_ = df.loc[val_idx,:].reset_index(drop=True)

    train_ds = PSMDataset(train_, data_root, transforms=get_train_transforms(), output_label=True, 
                          one_hot_label=False, do_fmix=True, do_cutmix=False)
    valid_ds = PSMDataset(valid_, data_root, transforms=get_valid_transforms(), output_label=True)
    
    device = torch.device(CFG['device'])

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=CFG['train_bs'],
        pin_memory=True,
        drop_last=False,
        shuffle=True,
        num_workers=CFG['num_workers'],
        #collate_fn=lambda x: default_collate(x).to(device)
        #sampler=BalanceClassSampler(labels=train_['label'].values, mode="downsampling")
    )
    val_loader = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=CFG['valid_bs'],
        num_workers=CFG['num_workers'],
        shuffle=False,
        pin_memory=True,
        #collate_fn=lambda x: default_collate(x).to(device)
    )

    return train_loader, val_loader
