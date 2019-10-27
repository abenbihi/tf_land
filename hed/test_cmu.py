
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
    ROOT_DATA_DIR='/mnt/dataX/assia/Extended-CMU-Seasons/'
else:
    print('Get your mtf machine cst correct you fool !')
    exit(1)

W, H = 1024, 768
#level_l = ['dsn%d'%d for d in range(1,6)]
#level_l += ['fuse']
level_l = ['fuse']


def eval_network(img_fn_l, out_fn_l):
    # prepare output dirs
    out_dir = 'res/cmu'
    
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
                
                for idx, img_fn in enumerate(img_fn_l):
                    if idx%20 == 0:
                        print('%d/%d'%(idx, len(img_fn_l)) )
                    img = cv2.imread(img_fn)
                    
                    # img pre-proc
                    img = img.astype(np.float32)
                    img -= MEAN_VGG 
                    img = np.expand_dims(img, 0)

                    if os.path.exists(out_fn_l[idx]):
                        continue

                    pred_np = sess.run( [preds_op["fuse"]], feed_dict={img_op: img})
                    # add the prediction to score_pred and increase count by 1
                    pred_np = np.squeeze(pred_np)
                    pred_np = expit(pred_np)
                    pred_np = (pred_np*255).astype(np.uint8)
                    #print('out_fn: %s'%out_fn)
                    cv2.imwrite(out_fn_l[idx], pred_np)
                
                #summary = tf.Summary()
                #summary.ParseFromString(sess.run(summary_op))
                #summary_writer.add_summary(summary, global_step)
            except Exception as e:  # pylint: disable=broad-except
                coord.request_stop(e)

            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)

def main(args):
    if args.survey_id == -1:
        meta_fn = 'meta/cmu/surveys/%d/c%d_db.txt'%(args.slice_id, args.cam_id)
    else:
        meta_fn = 'meta/cmu/surveys/%d/c%d_%d.txt'%(args.slice_id, args.cam_id, args.survey_id)
    
    # input
    meta = np.loadtxt(meta_fn, dtype=str)
    img_fn_l = ["%s/%s"%(args.img_dir, l) for l in meta[:,0]]
    root_fn_l = ["%s.png"%(l.split("/")[-1]).split(".")[0] for l in meta[:,0]]
    
    # output
    if not os.path.exists("%s/fuse/"%args.res_dir):
        os.makedirs("%s/fuse"%args.res_dir)

    out_fn_l = ["%s/fuse/%s"%(args.res_dir, l) for l in root_fn_l]
    print(img_fn_l[:3])
    print(out_fn_l[:3])
    eval_network(img_fn_l, out_fn_l)


if __name__ == '__main__':  
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, help='full path to subject data')
    parser.add_argument('--res_dir', type=str)
    parser.add_argument('--slice_id', type=int)
    parser.add_argument('--cam_id', type=int)
    parser.add_argument('--survey_id', type=int)
    parser.add_argument('--mean_file', type=str, help='path to mean file')
    parser.add_argument('--moving_average_decay', type=float, default=0.9999, help='')
    args = parser.parse_args()
    
    main(args)

