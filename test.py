# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 20:12:51 2020

@author: 86150
"""

import pickle
import numpy as np
#anno = pickle.load(open('C:/Users/86150/.spyder-py3/projects/TIN_bin/lib/ult/gt_hoi_py2/hoi_160.pkl', 'rb'))
#print(anno)

x=[1,2,3,4]
y=[5,6,7,8]
hbox = np.array(x)
obox = np.array(y)
z=np.concatenate((hbox, obox), axis=0)
print(z)