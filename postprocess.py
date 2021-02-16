#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd 

#fileneame = input('file name before .csv :')

ans = pd.read_csv('filesblend_sqrt_[4]_[1].csv', names=['label'])
anslist = list(ans.label.values)
print(ans.value_counts())
#print(anslist)


for idx, lb in enumerate(anslist): 

    # in N combination
    if anslist[idx:idx+2] == ['N1','N3']:
        ans.loc[idx+1]['label'] = 'N2'

    if anslist[idx:idx+5] == ['N2', 'N2', 'REM', 'N2', 'N2']:
        ans.loc[idx+2]['label'] = 'N2'
    if anslist[idx:idx+5] == ['N3', 'N3', 'REM', 'N3', 'N3']:
        ans.loc[idx+2]['label'] = 'N3'

    # in REM
    if anslist[idx:idx+3] == ['N1', 'REM', 'REM']:
        ans.loc[idx+0]['label'] = 'REM'
    if anslist[idx:idx+3] == ['N2', 'REM', 'REM']:
        ans.loc[idx+0]['label'] = 'REM'
    
    if anslist[idx:idx+3] == ['REM', 'N1', 'REM']:
        ans.loc[idx+1]['label'] = 'REM'
    if anslist[idx:idx+3] == ['REM', 'N2', 'REM']:
        ans.loc[idx+1]['label'] = 'REM'

    
print(ans.value_counts())
ans['label'].to_csv('files_post.csv', header=False, index=False)
print('files_post.csv is made...!')
