#!/usr/bin/env python
# -*- coding: utf-8 -*-

from loaderViTrgb import prepare_dataloader, seed_everything, str2int

from glob import glob
from sklearn.model_selection import GroupKFold, StratifiedKFold
import numpy as np
import pandas as pd
import csv
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.nn.modules.loss import _WeightedLoss

import time
import random
import os
import timm
import tqdm
from tqdm import tqdm
import torchvision.models as models
from sklearn.metrics import f1_score

import adamp 
from adamp import AdamP

package_paths = [
    '/FMix', '/packages/pytorch-image-models-master/',
    #'/packages/EfficientNet-PyTorch-master',
    '/packages'
]
import sys;

for pth in package_paths:
    sys.path.append(pth)

MODEL_PATH = './packages/jx_vit_base_p16_224-80ecf9dd.pth'

'''
'''
subject_list = pd.read_csv("/DATA/trainset-for_user.csv", names = ['ID', 'IDnum', 'label'])
subject_list['label'] = subject_list['label'].apply(str2int)

CFG = {
    'fold_num': 5,
    'seed': 72,
    'model_arch': 'vit_base_patch16_224', ##no pretrained
    'img_size': 224,  ##224 ->272
    'epochs': 100,
    'train_bs': 32,
    'valid_bs': 32,
    'T_0': 10,
    'lr': 1e-4,
    'min_lr': 1e-6,
    'weight_decay':1e-6,
    'num_workers': 4,
    'accum_iter': 2, # suppoprt to do batch accumulation for backprop with effectively larger batch size
    'verbose_step': 1,
    'device': 'cuda:0'
}
'''
'''
#seed_everything(seed=42)
'''
class FFTConv(nn.Module):   # Our customized neural network with FFT 
    def __init__(self, in_channels, out_channels, dilation):
        super(FFTConv, self).__init__()
        self.fftconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=(1, dilation), dilation=(1, dilation)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=(1, dilation), dilation=(1, dilation)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        return self.fftconv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d((1,2)),
            FFTConv(in_channels, out_channels, 1)
        )
    def forward(self, x):
        return self.down(x)

class DownFinal(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownFinal, self).__init__()
        self.downfinal = nn.Sequential(
            nn.MaxPool2d(2),
            FFTConv(in_channels, out_channels, 1)
        )
    def forward(self, x):
        return self.downfinal(x)

class FreqNetO(nn.Module):
    def __init__(self, channel, classes, hiddench):
        super(FreqNetO, self).__init__()
        self.channel = channel 
        self.classes = classes 
        self.fft1 = FFTConv(channel, hiddench, 1)
        self.fft2 = FFTConv(channel, hiddench, 2)
        self.fft3 = FFTConv(channel, hiddench, 3)
        self.fft4 = FFTConv(channel, hiddench, 4)
        self.fft5 = FFTConv(channel, hiddench, 5)
        self.fft6 = FFTConv(channel, hiddench, 6)
        self.fft7 = FFTConv(channel, hiddench, 7)
        self.fft8 = FFTConv(channel, hiddench, 8)
        self.down1 = Down(hiddench*8, hiddench*8)
        self.down2 = Down(hiddench*8, hiddench*16)
        self.down3 = DownFinal(hiddench*16, hiddench*16)

        self.fc = nn. Sequential(
            nn.Linear(hiddench*16*60*135, 125),
            nn.BatchNorm1d(125),
            nn.ReLU(),
            nn.Linear(125, 5)
        )
        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        x1, x2, x3, x4 = self.fft1(x), self.fft2(x), self.fft3(x), self.fft4(x)
        x5, x6, x7, x8 = self.fft5(x), self.fft6(x), self.fft7(x), self.fft8(x)
        x = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8), dim = 1)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        #print(x.size())
        final = self.soft(x)
        return final

def freqneto(channel=1, classes=5, hiddench=2):
    return FreqNetO(channel, classes, hiddench)

'''
class PSMClassifier(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=False):
        super().__init__()
        #self.model = models.densenet161(pretrained=False)
        #n_features = self.model.classifier.in_features
        
        #self.model = models.resnext50_32x4d(pretrained=False)
        #self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        #n_features = self.model.fc.in_features
        #self.model.classifier = nn.Linear(n_features, n_class)
        
        #self.model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=5, pretrained=False)
        #self.model = timm.creator_model('')
        self.model = timm.create_model(CFG['model_arch'], img_size = CFG['img_size'], in_chans=3,pretrained=False)
        if pretrained:
            self.model.load_state_dict(torch.load(MODEL_PATH))

        self.model.head = nn.Linear(self.model.head.in_features, n_class)
        '''
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            #nn.Linear(n_features, hidden_size,bias=True), nn.ELU(),
            nn.Linear(n_features, n_class, bias=True)
        )
        '''
    def forward(self, x):
        x = self.model(x)
        return x
'''
'''
def train_one_epoch(epoch, model, loss_fn, optimizer, train_loader, device, scheduler=None, schd_batch_update=False):
    model.train()

    t = time.time()
    running_loss = None

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()

        #print(image_labels.shape, exam_label.shape)
        with autocast():
            image_preds = model(imgs)   #output = model(input)
            #print(image_preds.shape, exam_pred.shape)
            
            #print(f1_loss_fn(image_preds, image_labels))

            loss = loss_fn(image_preds, image_labels)

            scaler.scale(loss).backward()

            if running_loss is None:
                running_loss = loss.item()
            else:
                running_loss = running_loss * .99 + loss.item() * .01

            if ((step + 1) %  CFG['accum_iter'] == 0) or ((step + 1) == len(train_loader)):
                # may unscale_ here if desired (e.g., to allow clipping unscaled gradients)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad() 
                
                if scheduler is not None and schd_batch_update:
                    scheduler.step()

            if ((step + 1) % CFG['verbose_step'] == 0) or ((step + 1) == len(train_loader)):
                description = f'epoch {epoch} loss: {running_loss:.4f}'
                
                pbar.set_description(description)
                
    if scheduler is not None and not schd_batch_update:
        scheduler.step()
        
def valid_one_epoch(epoch, model, loss_fn, val_loader, device, scheduler=None, schd_loss_update=False):
    model.eval()

    t = time.time()
    loss_sum = 0
    sample_num = 0
    image_preds_all = []
    image_targets_all = []
    
    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()
        
        image_preds = model(imgs)   #output = model(input)
        #print(image_preds.shape, exam_pred.shape)
        image_preds_all += [torch.argmax(image_preds, 1).detach().cpu().numpy()]
        image_targets_all += [image_labels.detach().cpu().numpy()]
        
        loss = loss_fn(image_preds, image_labels)
        
        loss_sum += loss.item()*image_labels.shape[0]
        sample_num += image_labels.shape[0]  

        if ((step + 1) % CFG['verbose_step'] == 0) or ((step + 1) == len(val_loader)):
            description = f'epoch {epoch} loss: {loss_sum/sample_num:.4f}'
            pbar.set_description(description)
    #f1 = F1(num_classes=5, average = 'macro')
    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)
    print('validation multi-class accuracy = {:.4f}'.format((image_preds_all==image_targets_all).mean()))
    print('validation multi-class f1 = {:.4f}'.format(f1_score(image_preds_all, image_targets_all, average='macro')))

    if scheduler is not None:
        if schd_loss_update:
            scheduler.step(loss_sum/sample_num)
        else:
            scheduler.step()


class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets:torch.Tensor, n_classes:int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                    device=targets.device) \
                .fill_(smoothing /(n_classes-1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
        return targets

    def forward(self, inputs, targets):
        targets = SmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
            self.smoothing)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss


if __name__ == "__main__":
    #train_val()
    #test() 

    seed_everything(CFG['seed'])
    
    folds = StratifiedKFold(n_splits=CFG['fold_num'], shuffle=True, random_state=CFG['seed']).split(np.arange(subject_list.shape[0]), subject_list.label.values)
    
    for fold, (trn_idx, val_idx) in enumerate(folds):
        if fold > 0:
            break 

        print('Training with {} started'.format(fold))

        print(len(trn_idx), len(val_idx))
        train_loader, val_loader = prepare_dataloader(subject_list, trn_idx, val_idx, data_root='/DATA')

        device = torch.device(CFG['device'])
        
        #train_loader, val_loader  = train_loader.to(device), val_laoder.to(device)
        model = PSMClassifier(CFG['model_arch'], subject_list.label.nunique(), pretrained=True).to(device)
        #model = freqneto(1, 5, 2).to(device)
        scaler = GradScaler()   
        
        optimizer = torch.optim.Adam(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
        #optimizer = torch.optim.Adam(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=CFG['epochs']-1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CFG['T_0'], T_mult=1, eta_min=CFG['min_lr'], last_epoch=-1)
        #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=25, 
        #                                                max_lr=CFG['lr'], epochs=CFG['epochs'], steps_per_epoch=len(train_loader))
        
        loss_tr = SmoothCrossEntropyLoss(smoothing=0.1).to(device)
        loss_fn = SmoothCrossEntropyLoss(smoothing=0.1).to(device)
        
        for epoch in range(CFG['epochs']):
            train_one_epoch(epoch, model, loss_tr, optimizer, train_loader, device, scheduler=scheduler, schd_batch_update=False)

            with torch.no_grad():
                valid_one_epoch(epoch, model, loss_fn, val_loader, device, scheduler=scheduler, schd_loss_update=False)

            torch.save(model.state_dict(),'{}_nf_fold_{}_{}'.format(CFG['model_arch'], fold, epoch))
            
        #torch.save(model.cnn_model.state_dict(),'{}/cnn_model_fold_{}_{}'.format(CFG['model_path'], fold, CFG['tag']))
        del model, optimizer, train_loader, val_loader, scaler, scheduler
        torch.cuda.empty_cache()
