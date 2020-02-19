# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 16:37:16 2020

@author: 86150
"""

import numpy as np

Human=[100,200,300,400]
ratio=0.85

Human    = np.round(np.array(Human).astype(np.float32) * ratio)

print(Human)