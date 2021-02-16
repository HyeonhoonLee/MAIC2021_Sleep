#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from loaderViTrgb import str2int
from inferblend import int2str

CFG = {
    'model_arch': 'vit_base_patch16_224',
    'model_arch2': 'tf_efficientnet_b4_ns', #'vit_base_patch16_224', #'tf_efficientnet_b4_ns',
    'used_epochs': [4],
    'used_epochs2': [1],
    'ens_weight': 0.06,  # 0.05
    'power': 0.25,
    'method' : 'sqrt'
}
fold = 0
test_list = pd.read_csv("/DATA/testset-for_user.csv", names= ['ID', 'IDnum'])



with open ('{}_fold_{}_{}.npy'.format(CFG['model_arch'], fold, CFG['used_epochs']), 'rb') as f: 
    tst_preds1 = np.load(f, allow_pickle=False)

with open ('{}_fold_{}_{}.npy'.format(CFG['model_arch2'], fold, CFG['used_epochs2']), 'rb') as f2: 
    tst_preds2 = np.load(f2, allow_pickle=False)



if CFG['method'] == 'avg':
## Average probablities
    ens_w = CFG['ens_weight']
    tst_preds = np.average((tst_preds1, tst_preds2), axis=0, weights=(ens_w, (1-ens_w)))

elif CFG['method'] == 'rank': 
    array1 = np.array(tst_preds1)
    order1 = array1.argsort()
    ranks1 = order1.argsort()

    array2 = np.array(tst_preds2)
    order2 = array2.argsort()
    ranks2 = order2.argsort()
    tst_preds  = np.sum((ranks1, ranks2), axis=0)

elif CFG['method'] == 'sqrt': #best weight = 0.06
    ens_w = CFG['ens_weight']
    array1 = np.sqrt(tst_preds1)
    array2 = np.sqrt(tst_preds2)
    tst_preds = array1 * ens_w + array2 * (1-ens_w)

elif CFG['method'] == 'log1p':
    ens_w = CFG['ens_weight']
    array1 = np.log1p(tst_preds1)
    array2 = np.log1p(tst_preds2)
    tst_preds = array1 * ens_w + array2 * (1-ens_w)

elif CFG['method'] == 'square':
    ens_w = CFG['ens_weight']
    power = CFG['power']
    array1 = tst_preds1 ** power
    array2 = tst_preds2 ** power
    tst_preds = array1 * ens_w + array2 * (1-ens_w)

elif CFG['method'] == 'expm1':
    ens_w = CFG['ens_weight']
    array1 = np.expm1(tst_preds1)
    array2 = np.expm1(tst_preds2)
    tst_preds = array1 * ens_w + array2 * (1-ens_w)

else:
    print('retry!')

test_list['label'] = np.argmax(tst_preds, axis=1)
test_list['label'] = test_list['label'].apply(int2str)

filename = 'filesblend_{}_{}_{}.csv'.format(CFG['method'], CFG['used_epochs'], CFG['used_epochs2'])

test_list.to_csv(filename, columns=['label'], header=False, index=False)

print('{} is saved...!'.format(filename))

