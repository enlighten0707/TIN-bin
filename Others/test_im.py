# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 15:59:28 2020

@author: 86150
"""
import pickle
import cv2
import numpy as np

image_folder_list = {'hico-train': '/Disk1/yonglu/iCAN/Data/hico_20160224_det/images/train2015/', 
                     'hico-test': '/Disk1/yonglu/iCAN/Data/hico_20160224_det/images/test2015/',
                     'hcvrd': '/Disk2/yonglu/behaviour_universe/hcvrd/image/', 
                     'openimage': '/Disk2/yonglu/behaviour_universe/openimage/image/',
                     'vcoco': '/Disk1/yonglu/iCAN/Data/v-coco/coco/images/all/', 
                     'pic': '/Disk2/yonglu/behaviour_universe/pic/All/', 
                     'long_tail_1': '/Disk2/yonglu/behaviour_universe/long_tail_round_1/long_tail_final/img_all/all/', 
                     'long_tail_2': '/Disk2/yonglu/behaviour_universe/long_tail_round_2/img_all/'}

def changeForm(Trainval_GT,interval_divide=5):
    GT_new=[]
    for key,value in Trainval_GT.items():
        count = 0
        length = len(value)
        while count < length:
            temp = value[count: min(count + interval_divide, length)]
            count += len(temp)
            GT_new.append(temp)
    return GT_new

image_id=100

Trainval_GT=pickle.load(open('/Disk1/yonglu/detectron2/new_boxes/Trainval_GT_10w_new.pkl', "rb"))
GT_new=changeForm(Trainval_GT)
GT       = GT_new[image_id] #image_id is int number
image_name = GT[0][0]  #image_name is a string
image_dataset=GT[0][4] 

im_file = image_folder_list[image_dataset] + image_name #new path added
im_tmp   = cv2.imread(im_file)
im_tmp_shape=im_tmp.shape
im       = cv2.resize(im_tmp, (512,np.round(im_tmp_shape[1]*512/im_tmp_shape[0])), interpolation = cv2.INTER_CUBIC)
im_orig  = im.astype(np.float32, copy=True)
im_orig -= np.array([[[102.9801, 115.9465, 122.7717]]])
im_shape = im_orig.shape
im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)
print(im_shape)
