# --------------------------------------------------------
# Tensorflow TIN
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ult.config import cfg
from ult.timer import Timer
from ult.ult import Get_next_sp_with_pose
from ult.binary_utils import obj_range, obj_name, Generate_NIS_map

import cv2
import cPickle as pickle
import numpy as np
import os
import sys
import glob
import time

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

def get_blob(image_id):
    im_file  = cfg.DATA_DIR + '/' + 'hico_20160224_det/images/test2015/HICO_test2015_' + (str(image_id)).zfill(8) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)
    return im_orig, im_shape

def getSigmoid(b,c,d,x,a=6):
    e = 2.718281828459
    return a/(1+e**(b-c*x))+d

def im_detect(sess, net, image_id, Test_RCNN, keys, binary, bboxes):

    # save image information
    This_image = []

    im_orig, _ = get_blob(image_id) 
    blobs = {}

    for Human_out in Test_RCNN[image_id]:
        
        if (Human_out[1] == 'Human') and (np.max(Human_out[5]) > 0.8): # This is a valid human
            
            blobs['H_num']   = 0
            blobs['H_boxes'] = [np.empty((0, 5), np.float64)]
            blobs['O_boxes'] = [np.empty((0, 5), np.float64)]
            blobs['sp']      = [np.empty((0, 64, 64, 3), np.float64)]
            blobs['gt_class_O'] = [np.empty((0, 80), np.float64)]

            H_box = np.array([0, Human_out[2][0],  Human_out[2][1],  Human_out[2][2],  Human_out[2][3]]).reshape(1,5)
            index = []
            for i in range(len(Test_RCNN[image_id])):
                Object = Test_RCNN[image_id][i]
                if (np.max(Object[5]) > 0.3) and not (np.all(Object[2] == Human_out[2])): # This is a valid object
                    # 1.the object detection result should > thres  2.the bbox detected is not an object
                    blobs['H_boxes'].append(H_box)
                    blobs['O_boxes'].append(np.array([0, Object[2][0],  Object[2][1],  Object[2][2],  Object[2][3]]).reshape(1,5))
                    blobs['sp'].append(Get_next_sp_with_pose(Human_out[2], Object[2], Human_out[6]).reshape(1, 64, 64, 3))
                    classid = Object[4] - 1
                    tmp = np.zeros((1, 80), np.float64)
                    tmp[0, classid] = 1
                    blobs['gt_class_O'].append(tmp)
                    blobs['H_num'] += 1
                    index.append(i)
            blobs['H_boxes'] = np.concatenate(blobs['H_boxes'], axis=0)
            blobs['O_boxes'] = np.concatenate(blobs['O_boxes'], axis=0)
            blobs['sp']      = np.concatenate(blobs['sp'], axis=0)
            blobs['gt_class_O'] = np.concatenate(blobs['gt_class_O'], axis=0)

            if blobs['H_num'] == 0:
                continue

            cls_prob_binary = net.test_image_binary(sess, im_orig, blobs)[0]
            
            for i in range(blobs['H_num']):
                Object = Test_RCNN[image_id][index[i]]
                classid = Object[4] - 1
                keys[classid].append(image_id)
                binary[classid].append(cls_prob_binary[i, :])
#                hbox = np.array(Human_out[2]).reshape(1, -1)
#                obox = np.array(Object[2]).reshape(1, -1)
#                bboxes[classid].append(np.concatenate([hbox, obox], axis=1))
                hbox = np.array(Human_out[2])
                obox = np.array(Object[2])
                bboxes[classid].append(np.concatenate((hbox, obox), axis=0))


def test_net(sess, net, Test_RCNN, output_dir):


    np.random.seed(cfg.RNG_SEED)
    bboxes, binary, keys = [], [], []
    
    for i in range(80):
        keys.append([])
        binary.append([])
        bboxes.append([])
    
    count = 0
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    for line in glob.iglob(cfg.DATA_DIR + '/' + 'hico_20160224_det/images/test2015/*.jpg'):

        _t['im_detect'].tic()
 
        image_id   = int(line[-9:-4])

        im_detect(sess, net, image_id, Test_RCNN, keys, binary, bboxes)

        _t['im_detect'].toc()
        print('im_detect: {:d}/{:d} {:.3f}s'.format(count + 1, 9658, _t['im_detect'].average_time))
        count += 1

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    map, mrec   = Generate_NIS_map(bboxes, binary, keys, output_dir)

    pickle.dump({'ap':map, 'rec':mrec}, open(output_dir + '/eval_nis.pkl', 'wb'))

    modes = ['pos_', 'neg_', 'pos-neg_']
    for i in range(3):
        with open(output_dir + '/' + modes[i] + 'result.txt', 'w') as f:
            f.write('%14s, ap: %.4f rec: %.4f \n' % ('Summary_' + modes[i], float(np.mean(map[i])), float(np.mean(mrec[i]))))
            for j in range(80):
                f.write('%14s, ap: %.4f rec: %.4f \n' % (obj_name[j], float(map[i, j]), float(mrec[i, j])))
        print("%14s, mAP: %.4f, mRec: %.4f" % (modes[i][:-1], float(np.mean(map[i])), float(np.mean(mrec[i]))))


