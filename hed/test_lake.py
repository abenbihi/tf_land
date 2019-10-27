
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import time
import sys
import argparse
from datetime import datetime

import cv2
import numpy as np
from scipy.special import expit
from sklearn.metrics import (precision_recall_curve, 
        precision_recall_fscore_support, roc_curve, auc) 

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf

import model
import tools

#W_PATH = '/home/gpu_user/assia/ws/tf/caffe-tensorflow/hed/data.ckpt'

W_PATH = 'meta/data.ckpt'
PAD = 16
MEAN_VGG = np.loadtxt('meta/mean_vgg.txt')

MACHINE = 1
if MACHINE==0:
    ROOT_DATA_DIR='/mnt/data_drive/dataset/lake/Dataset/'
elif MACHINE==1:
    ROOT_DATA_DIR='/mnt/lake/Dataset/'
else:
    print('Get your mtf machine cst correct you fool !')
    exit(1)

W, H = 704, 480
SUBDIR = {}
SUBDIR['2015'] = {}
SUBDIR['2015']['winter'] = '150216'
SUBDIR['2015']['spring'] = '150429'
SUBDIR['2015']['summer'] = '150723'
SUBDIR['2015']['autumn'] = '151027'

SUBDIR['2016'] = {}
SUBDIR['2016']['winter'] = '160216'
SUBDIR['2016']['spring'] = '160620'
SUBDIR['2016']['summer'] = '160829'
SUBDIR['2016']['autumn'] = '161114'

level_l = ['dsn%d'%d for d in range(1,6)]
level_l += ['fuse']


def eval_network(args):
    """Run Eval once.

    Args:
      saver: Saver.
      summary_writer: Summary writer.
      top_k_op: Top K op.
      summary_op: Summary op.
    """


    # prepare output dir
    #year, weather, seq = '2016', 'summer', 19
    year, weather, seq = '2015', 'summer', 22
    #year, weather, seq = '2016', 'summer', 5
    data_dir = '%s/%s/%s/%04d'%(ROOT_DATA_DIR, year,
        SUBDIR[year][weather], seq)

    out_dir = 'res/lake/%s/%s/%04d'%(year, weather, seq)
    if not os.path.exists(out_dir):
       os.makedirs(out_dir)
   
    
    with tf.Graph().as_default():
        img_op = tf.placeholder(dtype=tf.float32, shape=[1, H, W, 3])
        #tf.summary.image('img', img_op)
        preds_op = model.model(img_op) #compute predictions

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()
        graph_def = tf.get_default_graph().as_graph_def()
        summary_writer = tf.summary.FileWriter(out_dir, graph_def=graph_def)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            ### begin var  assignment OK !!! Do this one !
            if sys.version_info[0] >= 3:
                caffe_weights = np.load(W_PATH, encoding = 'latin1').item() # fuck pickles
            else:
                caffe_weights = np.load(W_PATH).item()
            var_to_restore = []
            hed_var_list = [l.split("\n")[0] for l in open('meta/hed_vars.txt').readlines()]
            var_list =  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            for var in var_list:
                if var.op.name in hed_var_list:
                    var_to_restore.append(var)
                    #print(var.op.name)
                    if len(var.op.name.split("/"))!=3:
                        print('No load %s' %var.op.name)
                        continue
                    scope, dummy, var_type = var.op.name.split("/")
                    #print('scope: %s, dummy: %s, var_type: %s' %(scope, dummy, var_type))
                    if 'conv2d_transpose' in dummy:
                        # TODO: I am not sure of the transpose 2,3 yet
                        #sess.run(var.assign(caffe_weights[scope][0].transpose(2,3,0,1)))
                        sess.run(var.assign(caffe_weights[scope][0].transpose(3,2,0,1)))
                    else:
                        if var_type=='kernel':
                            sess.run(var.assign(caffe_weights[scope]['weights']))
                        else:
                            sess.run(var.assign(caffe_weights[scope]['biases']))
            global_step = 0
            
            # Start the queue runners.
            coord = tf.train.Coordinator()
            try:
                threads = []
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
   

                img_fn_l = sorted(os.listdir(data_dir))
                img_num = len(img_fn_l)
                
                for img_id in range(549,img_num, 20):

                    img_fn = '%s/%s'%(data_dir, img_fn_l[img_id])
                    print('%s'%img_fn)
                    img = cv2.imread(img_fn)[:,:W]
                    print('raw_img_shape: ', img.shape)
                    cv2.imshow('img', img)
                    cv2.waitKey(1)
                    
                    # img pre-proc
                    img = img.astype(np.float32)
                    img -= MEAN_VGG 
                    img = np.expand_dims(img, 0)

                    pred_d = {}
                    for level in level_l:
                    
                        pred_np = sess.run( [preds_op[level]], feed_dict={img_op: img})
                            
                        # add the prediction to score_pred and increase count by 1
                        pred_np = np.squeeze(pred_np)
                        pred_np = expit(pred_np)
                        cv2.imshow(level, pred_np)
                        cv2.waitKey(1)

                    cv2.waitKey(0)
                        
                    # save img results
                    WRITE = (0==1)
                    if WRITE:
                        score_pred = (score_pred*255).astype(np.uint8)
                        seq = img_root_fn.split("/")[0]
                        out_fn = '%s/out/%s/%s'%(eval_log_dir, seq, img_root_fn.split("/")[-1])
                        cv2.imwrite(out_fn, score_pred)
                
                summary = tf.Summary()
                summary.ParseFromString(sess.run(summary_op))
                summary_writer.add_summary(summary, global_step)
            except Exception as e:  # pylint: disable=broad-except
                coord.request_stop(e)

            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)


if __name__ == '__main__':  
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, help='full path to subject data')
    parser.add_argument('--mean_file', type=str, help='path to mean file')
    parser.add_argument('--moving_average_decay', type=float, default=0.9999, help='')
    args = parser.parse_args()
    
    eval_network(args)


