"""A binary to train using a single GPU.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse
import os.path
import time
import math
from six.moves import xrange  # pylint: disable=redefined-builtin
from datetime import datetime

import numpy as np
from scipy.special import expit
import cv2

import tensorflow as tf
from tensorflow.contrib.framework.python.framework import checkpoint_utils

import model
from inputs import Dataloader

W_PATH = '/home/gpu_user/assia/ws/tf/caffe-tensorflow/hed/data.ckpt'

def train_network(args): 
    
    train_log_dir = os.path.join(args.log_dir, args.xp_name, 'train')

    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False) 
        
        # network inputs
        img_op = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, args.patch_h, args.patch_w, 3])
        gt_op = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, args.patch_h, args.patch_w, 1])

        # network model, loss, optimizer
        preds_op = model.model(img_op) #compute predictions
        loss_op = model.loss(preds_op['fuse'], gt_op) #compute loss
        train_op = model.train(loss_op, global_step, args) #train

        # Set saver to restore network before eval
        variable_averages = tf.train.ExponentialMovingAverage(
            args.moving_average_decay)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        

        # tensorboard log 
        tf.summary.image('img', img_op)
        tf.summary.image('gt', 255*gt_op)
        tf.summary.image('pred_fuse', 255*tf.sigmoid(preds_op['fuse']))
        tf.summary.image('pred_dsn1', 255*tf.sigmoid(preds_op['dsn1']))
        tf.summary.image('pred_dsn2', 255*tf.sigmoid(preds_op['dsn2']))
        tf.summary.image('pred_dsn3', 255*tf.sigmoid(preds_op['dsn3']))
        tf.summary.image('pred_dsn4', 255*tf.sigmoid(preds_op['dsn4']))
        tf.summary.image('pred_dsn5', 255*tf.sigmoid(preds_op['dsn5']))
 
        # (debug) load vgg weights
        #var_graph = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        #for var in var_graph:
        #    print(var.op.name)
               
        # Set summary op, restore vars
        summary_op = tf.summary.merge_all()
        
        # let's go
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            ### begin var  assignment OK !!! Do this one !
            if args.start == 1: # load initialization weights from caffe
                ### begin var  assignment OK !!! Do this one !
                caffe_weights = np.load(W_PATH).item()
                var_to_restore = []
                hed_var_list = [l.split("\n")[0] for l in open('meta/hed_vars.txt').readlines()]
                var_list =  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                for var in var_list:
                    #print(var.op.name)
                    #continue
                    if var.op.name in hed_var_list:
                        var_to_restore.append(var)
                        if len(var.op.name.split("/"))!=3:
                            print('No load %s' %var.op.name)
                            continue
                        scope, dummy, var_type = var.op.name.split("/")
                        #print('scope: %s, dummy: %s, var_type: %s' %(scope, dummy, var_type))
                        if 'conv2d_transpose' in dummy:
                            #sess.run(var.assign(caffe_weights[scope][0].transpose(2,3,0,1)))
                            sess.run(var.assign(caffe_weights[scope][0].transpose(3,2,0,1)))
                        else:
                            if var_type=='kernel':
                                sess.run(var.assign(caffe_weights[scope]['weights']))
                            else:
                                sess.run(var.assign(caffe_weights[scope]['biases']))
                global_step = 0
            else: # load trained weights from tensorflow
                ckpt = tf.train.get_checkpoint_state(train_log_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                else:
                    print('No checkpoint file found in: %s' %(train_log_dir))
                    return
            #### end var assignment
            
            # Start the queue runners. (it is a tf thing.)
            coord = tf.train.Coordinator()
            try:
                threads = []
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                  threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                             start=True))
            
                summary_writer = tf.summary.FileWriter(train_log_dir, graph_def=sess.graph_def)
                current_epoch = 0
                dataloader = Dataloader(args.img_dir, args.edge_dir, (1==1), 
                        args.batch_size, (args.patch_w, args.patch_h), args.mean_file)
                step = int(global_step)

                while current_epoch < args.epochs:
                    # train
                    next_epoch, img_batch, gt_batch = dataloader.next_batch()
                    #print(gt_batch.shape)
                    start_time = time.time()
                    #loss_value = 1
                    _, loss_value = sess.run([train_op, loss_op],
                            feed_dict={img_op: img_batch, gt_op: gt_batch})
                    
                    # debug
                    #toto = expit(np.squeeze(toto[0,:,:,:]))
                    #print(toto.shape)
                    #cv2.imshow('toto', (toto*255).astype(np.uint8))
                    #cv2.waitKey(0)
                            
                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                    duration = time.time() - start_time
                    step+=1
                    
                    # console log
                    if step % args.display_interval == 0:
                        num_examples_per_step = args.batch_size
                        examples_per_sec = num_examples_per_step / duration
                        sec_per_batch = float(duration)
                        format_str = ('%s: epoch %d/%d, step %d, loss=%.3f (%.1f examples/sec; %.3f ' 'sec/batch)')
                        print (format_str % (datetime.now(), current_epoch, args.epochs, step, 
                            loss_value, examples_per_sec, sec_per_batch))
                    
                    # tensorboard log
                    if (step % args.summary_interval==0):
                        summary_str = sess.run(summary_op, 
                                feed_dict={img_op: img_batch, gt_op: gt_batch})
                        summary_writer.add_summary(summary_str, step)
                    
                    # save model
                    if (step % args.save_interval==0):
                        checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=step)

                                               
                    if next_epoch==1:
                        current_epoch +=1

                # Save model
                summary_str = sess.run(summary_op,
                        feed_dict={img_op: img_batch, gt_op: gt_batch})
                summary_writer.add_summary(summary_str, step)
                checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                
            except Exception as e:  # pylint: disable=broad-except
                coord.request_stop(e)
            
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)


if __name__ == '__main__':  
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, help='full path to subject data')
    parser.add_argument('--edge_dir', type=str, help='full path to subject data')
    parser.add_argument('--mean_file', type=str, help='path to mean file')
    parser.add_argument('--patch_h', type=int, default=256, help='new height')
    parser.add_argument('--patch_w', type=int, default=256, help='new width')

    parser.add_argument('--xp_name', type=str, help='xp name')
    parser.add_argument('--log_dir', type=str, help='path to log dir')
    parser.add_argument('--display_interval', type=int, default=10, help='')
    parser.add_argument('--summary_interval', type=int, default=10, help='')
    parser.add_argument('--save_interval', type=int, default=1000000, help='')
    
    parser.add_argument('--epochs', type=int, help='num of epochs to train')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-03, help='learning rate')
    parser.add_argument('--adam_b1', type=float, default=0.9, help='adam beta1')
    parser.add_argument('--adam_b2', type=float, default=0.999, help='adam beta2')
    parser.add_argument('--adam_eps', type=float, default=1e-08, help='adam epsilon')
    parser.add_argument('--moving_average_decay', type=float, default=0.9999, help='')
    
    parser.add_argument('--start', type=int, default=1, help='Set to 1 to load hed weights')
    args = parser.parse_args()
    
    train_log_dir = os.path.join(args.log_dir, args.xp_name, 'train')
    if tf.gfile.Exists(train_log_dir):  
        tf.gfile.DeleteRecursively(train_log_dir)
    tf.gfile.MakeDirs(train_log_dir)

    train_network(args) #epochs_to_train, train_log_dir, data_dir, dataset_id)


