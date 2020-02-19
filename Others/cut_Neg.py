# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 18:14:23 2020

@author: 86150
"""

import pickle

Neg=pickle.load(open('Trainval_Neg_10w_new.pkl','rb'))

cnt=0
Neg_new={}

for key,value in Neg.items():
    if(cnt%5==0):
        Neg_new[key]=value
    cnt+=1
    if(cnt>=50000):
        break
    
pickle.dump(Neg_new, open('Trainval_Neg_10w_new_cut.pkl', 'wb'), protocol=2)

