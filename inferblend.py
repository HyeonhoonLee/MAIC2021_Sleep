#!/usr/batch_sizein/env python
# -*- coding: utf-8 -*-
from loaderViTrgb import PSMDataset, str2int, get_img, seed_everything, get_inference_transforms
from mainViTrgb import PSMClassifier
from mainEff import PSMClassifier2
from glob import glob
from sklearn.model_selection import GroupKFold, StratifiedKFold
import numpy as np
import pandas as pd
import cv2
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from multiprocessing import cpu_count
from PIL import Image
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)
from albumentations.pytorch import ToTensorV2

import time
import random
import os
import timm
import tqdm
from tqdm import tqdm
import sklearn
from sklearn.metrics import log_loss
from sklearn.metrics import f1_score
import torchvision.models as models

def int2str(lb):
    if lb == 0: lb = "Wake"
    elif lb == 1: lb = "N1"
    elif lb == 2: lb = "N2"
    elif lb == 3: lb = "N3"
    elif lb == 4: lb = "REM"
    return lb

subject_list = pd.read_csv("/DATA/trainset-for_user.csv", names = ['ID', 'IDnum', 'label'])
subject_list['label'] = subject_list['label'].apply(str2int)

test_list = pd.read_csv("/DATA/testset-for_user.csv", names = ['ID', 'IDnum'])
print(test_list.info())

CFG = {
    'fold_num': 5,
    'seed': 72,
    'model_arch': 'vit_base_patch16_224', #pretrained!
    'model_arch2': 'tf_efficientnet_b4_ns', # pretrained!
    'img_size': 224,
    'epochs': 10,
    'train_bs': 32,
    'valid_bs': 32,
    'lr': 1e-4,
    'num_workers': 4,
    'accum_iter': 21, # suppoprt to do batch accumulation for backprop with effectively larger batch size
    'verbose_step': 1,
    'device': 'cuda:0',
    'tta': 1,
    'used_epochs': [4],
    'used_epochs2': [1],
    'weights': [1,1,1,1],
    'img_h': 224,
    'img_w': 224,
    'ens_weight': 0.5 # weight for model1(=ViT)`
}
#seed_everything(seed=42)


def inference_one_epoch(model, data_loader, device):
    model.eval()

    image_preds_all = []

    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for step, (imgs) in pbar:
        imgs = imgs.to(device).float()

        image_preds = model(imgs)   #output = model(input)
        image_preds_all += [torch.softmax(image_preds, 1).detach().cpu().numpy()]


    image_preds_all = np.concatenate(image_preds_all, axis=0)
    return image_preds_all


if __name__ == "__main__":
    #train_val()
    #test()

    seed_everything(CFG['seed'])

    folds = StratifiedKFold(n_splits=CFG['fold_num'], shuffle=True, random_state=CFG['seed']).split(np.arange(subject_list.shape[0]), subject_list.label.values)

    for fold, (trn_idx, val_idx) in enumerate(folds):
        # we'll train fold 0 first
        if fold > 0:
            break

        print('Inference with {} started'.format(fold))

        valid_ = subject_list.loc[val_idx,:].reset_index(drop=True)
        valid_ds = PSMDataset(valid_, '/DATA', transforms=get_inference_transforms(), output_label=False)
        
        ##for tesing 
        #test_list = test_list.loc[:3000,:].reset_index(drop=True)

        test_ds = PSMDataset(test_list, '/DATA', transforms=get_inference_transforms(), output_label=False)

        val_loader = torch.utils.data.DataLoader(
            valid_ds,
            batch_size=CFG['valid_bs'],
            num_workers=CFG['num_workers'],
            shuffle=False,
            pin_memory=False,
        )

        tst_loader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=CFG['valid_bs'],
            num_workers=CFG['num_workers'],
            shuffle=False,
            pin_memory=False,
        )

        device = torch.device(CFG['device'])

        model = PSMClassifier(CFG['model_arch'], 5).to(device)
        model2 = PSMClassifier2(CFG['model_arch2'], 5).to(device)


        val_preds = []
        tst_preds = []

        for i, epoch in enumerate(CFG['used_epochs']):
            model.load_state_dict(torch.load('{}_fold_{}_{}'.format(CFG['model_arch'], fold, epoch))) # check directory


            with torch.no_grad():
                for _ in range(CFG['tta']):
                    val_preds += [CFG['weights'][i]/sum(CFG['weights'])/CFG['tta']*inference_one_epoch(model, val_loader, device)]
                    tst_preds += [CFG['weights'][i]/sum(CFG['weights'])/CFG['tta']*inference_one_epoch(model, tst_loader, device)]

        val_preds = np.array(val_preds)
        tst_preds = np.array(tst_preds)
        val_preds = np.mean(val_preds, axis=0)
        tst_preds = np.mean(tst_preds, axis=0)
        print(tst_preds.shape)
        
        with open('{}_fold_{}_{}.npy'.format(CFG['model_arch'], fold, CFG['used_epochs']), 'wb') as f:
            np.save(f, tst_preds, allow_pickle=False)
        print('prob_of_model is saved...!')
        

        val_preds2 = []
        tst_preds2 = []

        for i, epoch in enumerate(CFG['used_epochs2']):

            model2.load_state_dict(torch.load('{}_fold_{}_{}'.format(CFG['model_arch2'], fold, epoch))) # check directory

            with torch.no_grad():
                for _ in range(CFG['tta']):
                    val_preds2 += [CFG['weights'][i]/sum(CFG['weights'])/CFG['tta']*inference_one_epoch(model2, val_loader, device)]
                    tst_preds2 += [CFG['weights'][i]/sum(CFG['weights'])/CFG['tta']*inference_one_epoch(model2, tst_loader, device)]

        val_preds2 = np.array(val_preds2)
        tst_preds2 = np.array(tst_preds2)
        val_preds2 = np.mean(val_preds2, axis=0)
        tst_preds2 = np.mean(tst_preds2, axis=0)
        new_val_preds2 = np.zeros((val_preds2.shape[0], 5))
        new_tst_preds2 = np.zeros((tst_preds2.shape[0], 5))
        for art in range(200):
            new_val_preds2[:, :] += val_preds2[:, art*5:(art+1)*5]
            new_tst_preds2[:, :] += tst_preds2[:, art*5:(art+1)*5]
        val_preds2 = new_val_preds2 / 200
        tst_preds2 = new_tst_preds2 / 200

        print(tst_preds2.shape)
        with open('{}_fold_{}_{}.npy'.format(CFG['model_arch2'], fold, CFG['used_epochs2']), 'wb') as f:
            np.save(f, tst_preds2, allow_pickle=False)
        print('prob_of_model2 is saved...!')

        #final ensemble...!
        ens_w = CFG['ens_weight']
        #val_preds = np.average(np.array([val_preds, val_preds2]), axis=0, weights=(ens_w, (1-ens_w)))
        #tst_preds = np.average(np.array([tst_preds, tst_preds2]), axis=0, weights=(ens_w, (1-ens_w)))
        val_preds = val_preds * ens_w + val_preds2 * (1-ens_w)
        tst_preds = tst_preds * ens_w + tst_preds2 * (1-ens_w)
        print(tst_preds.shape)


        #print('fold {} validation loss = {:.5f}'.format(fold, log_loss(valid_.label.values, val_preds)))
        print('fold {} validation accuracy = {:.5f}'.format(fold, (valid_.label.values==np.argmax(val_preds, axis=1)).mean()))
        print('fold {} validation f1 = {:.5f}'.format(fold, f1_score(valid_.label.values, np.argmax(val_preds, axis=1), average='macro')))
        print('done!')
        del model
        del model2
        torch.cuda.empty_cache()
        
        test_list['label'] = np.argmax(tst_preds, axis=1)
        test_list['label'] = test_list['label'].apply(int2str)

        '''
        #if 'tst_preds2' in globals():
        test_list.to_csv('files_blend_ViT{}_Eff{}.csv'.format(CFG['used_epochs'], CFG['used_epochs2']), columns=['label'], header=False, index=False)
        test_list.to_csv('filesfull_blend_ViT{}_Eff{}.csv'.format(CFG['used_epochs'], CFG['used_epochs2']), columns=['ID', 'IDnum', 'label'], header=False, index=False )
        
        else:
            test_list.to_csv('files_{}_fold_{}_{}.csv'.format(CFG['model_arch'], fold, CFG['used_epochs']), columns=['label'], header=False, index=False)
            test_list.to_csv('filesfull_{}_fold_{}_{}.csv'.format(CFG['model_arch'], fold, CFG['used_epochs']), columns=['ID', 'IDnum', 'label'], header=False, index=False )
        '''
