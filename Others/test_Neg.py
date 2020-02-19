# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 20:14:16 2020

@author: 86150
"""

import pickle

Trainval_Neg=pickle.load(open('Trainval_Neg_10w_new.pkl','rb'))
cnt=0;
for key,value in Trainval_Neg.items():
    if(cnt==50):
        break;
    if(cnt%10==0):
        print(value)
    cnt=cnt+1;
        