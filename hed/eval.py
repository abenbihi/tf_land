"""
Compute validation on cropped img to monitor the training
"""
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

import tensorflow as tf

import model
from inputs import Dataloader

W_PATH = '/home/gpu_user/assia/ws/tf/caffe-tensorflow/hed/data.ckpt'

def eval_network(args):
    """Run Eval once.

    Args:
      saver: Saver.
      summary_writer: Summary writer.
      top_k_op: Top K op.
      summary_op: Summary op.
    """
    # prepare output dir
    train_log_dir = os.path.join(args.log_dir, args.xp_name, 'train')
    eval_log_dir = os.path.join(args.log_dir, args.xp_name, 'val')

    ##for seq in ['08', '09', '10']:
    #for seq in ['09']:
    #    out_dir = '%s/out/%s'%(eval_log_dir, seq)
    #    if not os.path.exists(out_dir):
    #        os.makedirs(out_dir)
    
    
    with tf.Graph().as_default():
        # graph input
        img_op = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, args.patch_h, args.patch_w, 3])
        gt_op = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, args.patch_h, args.patch_w, 1])
        
        # network model
        preds_op = model.model(img_op) #compute predictions
        loss_op = model.loss(preds_op['fuse'], gt_op) #compute loss

        # tensorboard log
        tf.summary.image('img', img_op)
        tf.summary.image('gt', gt_op)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(args.moving_average_decay)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()
        graph_def = tf.get_default_graph().as_graph_def()
        summary_writer = tf.summary.FileWriter(eval_log_dir, graph_def=graph_def)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(train_log_dir)
            print("checkpoint path: ", ckpt.model_checkpoint_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            else:
                print('No checkpoint file found')
                return

            
            # Start the queue runners.
            coord = tf.train.Coordinator()
            try:
                threads = []
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
                next_epoch = 0
                dataloader = Dataloader(args.img_dir, args.edge_dir, (0==1), 
                        args.batch_size, (args.patch_w, args.patch_h), args.mean_file)

                mean_loss, tp, mean_pr = 0,0,0
                step = 1

                while True:
                    next_epoch, img_batch, gt_batch = dataloader.next_batch()
                    if next_epoch==1:
                        break
                    pred_np, loss_np = sess.run([preds_op['fuse'], loss_op],
                            feed_dict={img_op: img_batch, gt_op: gt_batch})
                    pix_num_per_batch = np.prod(gt_batch.shape)

                    # update validation loss and count true positives i.e.
                    # recall
                    if step==1:
                        mean_loss = np.sum(loss_np)
                        label_pred = (pred_np>0.5).astype(np.int)
                        tp = np.sum(gt_batch==label_pred)
                        mean_pr = tp/pix_num_per_batch
                    else:
                        mean_loss = (mean_loss + np.sum(loss_np)/(step-1)) * ( (step-1)/step)
                        label_pred = (pred_np>0.5).astype(np.int)
                        tp = np.sum(gt_batch==label_pred)
                        mean_pr = (mean_pr + tp/(pix_num_per_batch*(step-1))) * ( (step-1)/step) 
                    step += 1


                print('mean_loss: %.3f - mean_pr: %.3f' %(mean_loss, mean_pr))

                summary = tf.Summary()
                summary.ParseFromString(sess.run(summary_op,
                    feed_dict={img_op: img_batch, gt_op: gt_batch}))
                summary.value.add(tag='Loss validation', simple_value=mean_loss)
                summary.value.add(tag='Precision @ 1', simple_value=mean_pr)
                summary_writer.add_summary(summary, global_step)
            except Exception as e:  # pylint: disable=broad-except
                coord.request_stop(e)

            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)


if __name__ == '__main__':  
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, help='full path to subject data')
    parser.add_argument('--edge_dir', type=str, help='full path to subject data')
    parser.add_argument('--mean_file', type=str, help='path to mean file')
    parser.add_argument('--patch_h', type=int, default=480, help='new height')
    parser.add_argument('--patch_w', type=int, default=704, help='new width')
    
    parser.add_argument('--xp_name', type=str, help='xp name')
    parser.add_argument('--log_dir', type=str, help='path to log dir')
    
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--moving_average_decay', type=float, default=0.9999, help='')
    args = parser.parse_args()
    
    train_log_dir = os.path.join(args.log_dir, args.xp_name, 'train')
    if not os.path.exists(train_log_dir):
        print('Error: train log dir does not exists: %s' %train_log_dir)
        exit(1)

    eval_log_dir = os.path.join(args.log_dir, args.xp_name, 'val')
    if tf.gfile.Exists(eval_log_dir):
        tf.gfile.DeleteRecursively(eval_log_dir)
    tf.gfile.MakeDirs(eval_log_dir)

    eval_network(args)


