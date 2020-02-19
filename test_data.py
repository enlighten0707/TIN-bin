# -*- coding: utf-8 -*-
import numpy as np
import pickle 

data=pickle.load(open('binary_score_all.pkl','rb'))
for i in range (10):
    print(data[i])

