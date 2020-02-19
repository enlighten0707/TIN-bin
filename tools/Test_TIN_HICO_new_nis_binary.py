# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Tensorflow TIN
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import tensorflow as tf
import numpy as np
import argparse
import cPickle as pickle
import json
import os

from networks.TIN_HICO_bin import ResNet50
from ult.config import cfg
from models.test_Solver_HICO_new_nis_binary import test_net

os.environ['CUDA_VISIBLE_DEVICES'] = '2' # use GPU 0,1
def parse_args():
    parser = argparse.ArgumentParser(description='Test TIN on HICO dataset')
    parser.add_argument('--iteration', dest='iteration',
            help='Specify which weight to load',
            default=1700000, type=int)
    parser.add_argument('--model', dest='model',
            help='Select model',
            default='TIN_HICO', type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()

    # test detections result
    Test_RCNN      = pickle.load( open( cfg.DATA_DIR + '/' + 'Test_Faster_RCNN_R-50-PFN_2x_HICO_DET_with_pose.pkl', "rb" ) ) 

    # pretrain model
    weight = cfg.ROOT_DIR + '/Weights/' + args.model + '/HOI_iter_' + str(args.iteration) + '.ckpt'
    print ('Iter = ' + str(args.iteration) + ', path = ' + weight ) 
    output_file = cfg.ROOT_DIR + '/-Results/' + args.model + '_binary_' + str(args.iteration) + '/'

    # init session

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True
    sess = tf.Session(config=tfconfig)

    net = ResNet50()
    net.create_architecture(False)

    saver = tf.train.Saver()
    saver.restore(sess, weight)

    print('Pre-trained weights loaded.')
    
    test_net(sess, net, Test_RCNN, output_file)
    sess.close()

