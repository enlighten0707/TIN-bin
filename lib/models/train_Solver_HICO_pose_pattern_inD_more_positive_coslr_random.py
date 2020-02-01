# --------------------------------------------------------
# Tensorflow TIN
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ult.config import cfg
from ult.ult import Get_Next_Instance_HO_Neg_HICO_pose_pattern_version2
from ult.timer import Timer

import pickle
import numpy as np
import os
import sys
import glob
import time
import ipdb

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.training.learning_rate_decay import cosine_decay_restarts


 
class SolverWrapper(object):
    """
    A wrapper class for the training process
    """

    def __init__(self, sess, network, Trainval_GT, Trainval_N, output_dir, tbdir, Pos_augment, Neg_select, Restore_flag, pretrained_model, interval_divide):

        self.net               = network
        self.Trainval_GT       = self.changeForm(Trainval_GT, interval_divide)
        self.Trainval_N        = Trainval_N
        self.output_dir        = output_dir
        self.tbdir             = tbdir
        self.Pos_augment       = Pos_augment
        self.Neg_select        = Neg_select
        self.Restore_flag      = Restore_flag
        self.pretrained_model  = pretrained_model

    def snapshot(self, sess, iter):

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Store the model snapshot
        filename = 'HOI' + '_iter_{:d}'.format(iter) + '.ckpt'
        filename = os.path.join(self.output_dir, filename)
        self.saver.save(sess, filename)
        print('Wrote snapshot to: {:s}'.format(filename))
    
#    def changeForm(self, Trainval_GT, interval_divide):
#        GT_dict = {}
#        for item in Trainval_GT:
#            try:
#                GT_dict[item[0]].append(item)
#            except KeyError:
#                GT_dict[item[0]] = [item]
#
#        GT_new = []
#        for image_id, value in GT_dict.items():
#            count = 0
#            length = len(value)
#            while count < length:
#                temp = value[count: min(count + interval_divide, length)]
#                count += len(temp)
#                GT_new.append(temp)
#
#        return GT_new
        
    def changeForm(self,Trainval_GT,interval_divide=5):
        GT_new=[]
        for key,value in Trainval_GT.items():
            count = 0
            length = len(value)
            while count < length:
                temp = value[count: min(count + interval_divide, length)]
                count += len(temp)
                GT_new.append(temp)
        return GT_new
    
    def construct_graph(self, sess):
        with sess.graph.as_default():
      
            # Set the random seed for tensorflow
            tf.set_random_seed(cfg.RNG_SEED)

            # Build the main computation graph
            layers = self.net.create_architecture(True) # is_training flag: True

            # Define the loss
            loss = layers['total_loss']
            
            path_iter = self.pretrained_model.split('.ckpt')[0]
            iter_num = path_iter.split('_')[-1]

            # from iter_ckpt
            if cfg.TRAIN_MODULE_CONTINUE == 1:
                global_step    = tf.Variable(int(iter_num), trainable=False)

            # from iter 0
            if cfg.TRAIN_MODULE_CONTINUE == 2:
                global_step    = tf.Variable(0, trainable=False)

            #lr             = tf.train.exponential_decay(cfg.TRAIN.LEARNING_RATE * 10, global_step, cfg.TRAIN.STEPSIZE * 5, cfg.TRAIN.GAMMA, staircase=True) 
            # here we use cos lr scheme, i.e. 
            first_decay_steps = 2 * len(self.Trainval_GT) # 2 epoches
            lr = cosine_decay_restarts(cfg.TRAIN.LEARNING_RATE * 10, global_step, first_decay_steps, t_mul=2.0, m_mul=1.0, alpha=0.0) 
            self.optimizer = tf.train.MomentumOptimizer(lr, cfg.TRAIN.MOMENTUM)

            # list_var_to_update = []
            # if cfg.TRAIN_MODULE_UPDATE == 1:
            #     list_var_to_update = tf.trainable_variables()
            # if cfg.TRAIN_MODULE_UPDATE == 2:
            #     list_var_to_update = [var for var in tf.trainable_variables() if 'fc_binary' in var.name or 'binary_classification' in var.name]

            # 1--Update_all_parameter, 2--Only_Update_D, 3--Update_H+O+SP, 4--updating except classifiers of S(fc)
            list_var_to_update = []
            if cfg.TRAIN_MODULE_UPDATE == 1:
                list_var_to_update = tf.trainable_variables()
            if cfg.TRAIN_MODULE_UPDATE == 2:
                list_var_to_update = [var for var in tf.trainable_variables() if 'fc_binary' in var.name or 'binary_classification' in var.name]
            if cfg.TRAIN_MODULE_UPDATE == 3:
                list_var_to_update = [var for var in tf.trainable_variables() if 'fc_binary' not in var.name or 'binary_classification' not in var.name]
            if cfg.TRAIN_MODULE_UPDATE == 4:
                list_var_to_update = [var for var in tf.trainable_variables() if 'classification' not in var.name]

            grads_and_vars = self.optimizer.compute_gradients(loss, list_var_to_update)
            capped_gvs     = [(tf.clip_by_norm(grad, 1.), var) for grad, var in grads_and_vars]
            
            train_op = self.optimizer.apply_gradients(capped_gvs,global_step=global_step)
            self.saver = tf.train.Saver(max_to_keep=cfg.TRAIN.SNAPSHOT_KEPT)
            # Write the train and validation information to tensorboard
            self.writer = tf.summary.FileWriter(self.tbdir, sess.graph)

        return lr, train_op



    def from_snapshot(self, sess):
    
        
        if self.Restore_flag == 0:

            saver_t  = [var for var in tf.model_variables() if 'conv1' in var.name and 'conv1_sp' not in var.name]
            saver_t += [var for var in tf.model_variables() if 'conv2' in var.name and 'conv2_sp' not in var.name]
            saver_t += [var for var in tf.model_variables() if 'conv3' in var.name]
            saver_t += [var for var in tf.model_variables() if 'conv4' in var.name]
            saver_t += [var for var in tf.model_variables() if 'conv5' in var.name]
            saver_t += [var for var in tf.model_variables() if 'shortcut' in var.name]

            sess.run(tf.global_variables_initializer())
            for var in tf.trainable_variables():
                print(var.name, var.eval().mean())

            print('Restoring model snapshots from {:s}'.format(self.pretrained_model))

            self.saver_restore = tf.train.Saver(saver_t)
            self.saver_restore.restore(sess, self.pretrained_model)

            for var in tf.trainable_variables():
                print(var.name, var.eval().mean())


        if self.Restore_flag == 5 or self.Restore_flag == 6 or self.Restore_flag == 7:

            sess.run(tf.global_variables_initializer())
            for var in tf.trainable_variables():
                print(var.name, var.eval().mean())
            
            
            print('Restoring model snapshots from {:s}'.format(self.pretrained_model))
            saver_t = {}
            
            # Add block0
            for ele in tf.model_variables():
                if 'resnet_v1_50/conv1/weights' in ele.name or 'resnet_v1_50/conv1/BatchNorm/beta' in ele.name or 'resnet_v1_50/conv1/BatchNorm/gamma' in ele.name or 'resnet_v1_50/conv1/BatchNorm/moving_mean' in ele.name or 'resnet_v1_50/conv1/BatchNorm/moving_variance' in ele.name:
                    saver_t[ele.name[:-2]] = ele
            # Add block1
            for ele in tf.model_variables():
                if 'block1' in ele.name:
                    saver_t[ele.name[:-2]] = ele
           
            # Add block2
            for ele in tf.model_variables():
                if 'block2' in ele.name:
                    saver_t[ele.name[:-2]] = ele
                    
            # Add block3
            for ele in tf.model_variables():
                if 'block3' in ele.name:
                    saver_t[ele.name[:-2]] = ele
                
            # Add block4
            for ele in tf.model_variables():
                if 'block4' in ele.name:
                    saver_t[ele.name[:-2]] = ele
            
            self.saver_restore = tf.train.Saver(saver_t)
            self.saver_restore.restore(sess, self.pretrained_model)
            
            if self.Restore_flag >= 5:

                saver_t = {}
                # Add block5
                for ele in tf.model_variables():
                    if 'block4' in ele.name:
                        saver_t[ele.name[:-2]] = [var for var in tf.model_variables() if ele.name[:-2].replace('block4','block5') in var.name][0]
         
                
                self.saver_restore = tf.train.Saver(saver_t)
                self.saver_restore.restore(sess, self.pretrained_model)
            

            if self.Restore_flag >= 6:
                saver_t = {}
                # Add block6
                for ele in tf.model_variables():
                    if 'block4' in ele.name:
                        saver_t[ele.name[:-2]] = [var for var in tf.model_variables() if ele.name[:-2].replace('block4','block6') in var.name][0]
         
                
                self.saver_restore = tf.train.Saver(saver_t)
                self.saver_restore.restore(sess, self.pretrained_model)
                
            if self.Restore_flag >= 7:

                saver_t = {}
                # Add block7
                for ele in tf.model_variables():
                    if 'block4' in ele.name:
                        saver_t[ele.name[:-2]] = [var for var in tf.model_variables() if ele.name[:-2].replace('block4','block7') in var.name][0]
         
            
                self.saver_restore = tf.train.Saver(saver_t)
                self.saver_restore.restore(sess, self.pretrained_model)


        for var in tf.trainable_variables():
            print(var.name, var.eval().mean())


    def from_previous_ckpt(self,sess):

        sess.run(tf.global_variables_initializer())
        for var in tf.trainable_variables(): # trainable weights, we need surgery
            print(var.name, var.eval().mean())

        print('Restoring model snapshots from {:s}'.format(self.pretrained_model))
        saver_t = {}

        saver_t  = [var for var in tf.model_variables() if 'fc_binary' not in var.name \
                                                       and 'binary_classification' not in var.name \
                                                       and 'conv1_pose_map' not in var.name \
                                                       and 'pool1_pose_map' not in var.name \
                                                       and 'conv2_pose_map' not in var.name \
                                                       and 'pool2_pose_map' not in var.name]

        self.saver_restore = tf.train.Saver(saver_t)
        self.saver_restore.restore(sess, self.pretrained_model)

        print("the variables is being trained now \n")
        for var in tf.trainable_variables():
           print(var.name, var.eval().mean())

    
    def from_best_trained_model(self, sess):

        sess.run(tf.global_variables_initializer())
        for var in tf.trainable_variables(): # trainable weights, we need surgery
            print(var.name, var.eval().mean())

        print('Restoring model snapshots from {:s}'.format(self.pretrained_model))
        saver_t = {}

        saver_t  = [var for var in tf.model_variables() if 'fc_binary' not in var.name \
                                           and 'binary_classification' not in var.name \
                                           and 'conv1_pose_map' not in var.name \
                                           and 'pool1_pose_map' not in var.name \
                                           and 'conv2_pose_map' not in var.name \
                                           and 'pool2_pose_map' not in var.name \
                                           and 'global_classification' not in var.name \
                                           and 'object_fusion' not in var.name \
                                           and 'fc_GHO' not in var.name]

        for var in tf.trainable_variables():
            print(var.name, var.eval().mean())

        # for ele in tf.model_variables():
        #     saver_t[ele.name[:-2]] = ele

        self.saver_restore = tf.train.Saver(saver_t)
        self.saver_restore.restore(sess, self.pretrained_model)


        print("the variables is being trained now \n")
        for var in tf.trainable_variables():
           print(var.name, var.eval().mean())


    def train_model(self, sess, max_iters):
    
        lr, train_op = self.construct_graph(sess)

        if cfg.TRAIN_MODULE_CONTINUE == 1:
            self.from_previous_ckpt(sess)

        else:
            if cfg.TRAIN_INIT_WEIGHT == 2:
                self.from_best_trained_model(sess)

            if cfg.TRAIN_INIT_WEIGHT == 1:
                self.from_snapshot(sess)  

            if cfg.TRAIN_INIT_WEIGHT == 3:  # load all paras including D, initial from our best
                self.from_best_trained_model(sess) 
    
        sess.graph.finalize()

        timer = Timer()

        path_iter = self.pretrained_model.split('.ckpt')[0]
        iter_num = path_iter.split('_')[-1]

        if cfg.TRAIN_MODULE_CONTINUE == 2:
            iter = 0

        if cfg.TRAIN_MODULE_CONTINUE == 1:
            iter = int(iter_num)

        Data_length = len(self.Trainval_GT)
        idx = range(Data_length)
        np.random.shuffle(idx)

        while iter < max_iters + 1:

            timer.tic()

            if iter % Data_length == 0:
                np.random.shuffle(idx)
            image_id = idx[iter % Data_length]

            blobs = Get_Next_Instance_HO_Neg_HICO_pose_pattern_version2(self.Trainval_GT, self.Trainval_N, image_id, self.Pos_augment, self.Neg_select, Data_length)

            if (iter % cfg.TRAIN.SUMMARY_INTERVAL == 0) or (iter < 20):
                # Compute the graph with summary
                total_loss, summary = self.net.train_step_with_summary(sess, blobs, lr.eval(), train_op)
                self.writer.add_summary(summary, float(iter))
            else:
                # Compute the graph without summary
                total_loss = self.net.train_step(sess, blobs, lr.eval(), train_op)

            timer.toc()

            # Display training information
            if iter % (cfg.TRAIN.DISPLAY) == 0:
                print('iter: %d / %d, im_id: %u, total loss: %.6f, lr: %f, speed: %.3f s/iter' % \
                      (iter, max_iters, image_id, total_loss, lr.eval(), timer.average_time))

            # Snapshotting
            if (iter % cfg.TRAIN.SNAPSHOT_ITERS * 5 == 0 and iter != 0) or (iter == 10):
                
                self.snapshot(sess, iter)

            iter += 1

        self.writer.close()



def train_net(network, Trainval_GT, Trainval_N, output_dir, tb_dir, Pos_augment, Neg_select, Restore_flag, pretrained_model, max_iters=300000):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)
        
    if cfg.TRAIN_MODULE_CONTINUE == 2:
        # Remove previous events
        filelist = [ f for f in os.listdir(tb_dir)]
        for f in filelist:
            os.remove(os.path.join(tb_dir, f))
        # Remove previous snapshots
        filelist = [ f for f in os.listdir(output_dir)]
        for f in filelist:
            os.remove(os.path.join(output_dir, f))                
        
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    interval_divide = 5

    with tf.Session(config=tfconfig) as sess:
        sw = SolverWrapper(sess, network, Trainval_GT, Trainval_N, output_dir, tb_dir, Pos_augment, Neg_select, Restore_flag, pretrained_model, interval_divide)
        
        print('Solving..., Pos augment = ' + str(Pos_augment) + ', Neg augment = ' + str(Neg_select) + ', Restore_flag = ' + str(Restore_flag))
        sw.train_model(sess, max_iters)
        print('done solving')
