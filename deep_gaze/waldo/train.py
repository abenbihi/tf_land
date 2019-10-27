"""A binary to train using a single GPU.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import os.path
import argparse
from datetime import datetime
from six.moves import xrange  # pylint: disable=redefined-builtin

import math
import numpy as np
from scipy.misc import logsumexp

import tensorflow as tf
from tensorflow.contrib.framework.python.framework import checkpoint_utils


from tools.tf_tools import init_weights
from tools.tf_tools import fucking_deep_gaze_logsumexp as tf_logsumexp
from tools.cst import *
from waldo.inputs import Dataloader
import waldo.model as model
import waldo.loss as loss

if MACHINE!=2:
    import cv2

def train_network(args): 
    """

    Args:
        model: network model
        epochs_to_train: Number of epochs before eval
        train_log_dir: 
        data_dir: 
    """
    train_log_dir = os.path.join(args.train_log_dir, args.xp_name)
    is_training = True
    readout_net_num = 10
    convT_filter_np = np.load('meta/proto/resize_feature-upscale-conv2d_transpose-filter.npy')
    mean_np = np.array([103.94, 116.78, 123.68])
    
    # data
    csv_file = os.path.join(args.dataset_dir, args.dataset_id, 'train.txt')
    dataloader = Dataloader(csv_file, (args.dataset_dir+'/'+args.dataset_id), args.batch_size, 
            (args.w, args.h), args.resize_img, (1==1))
    
    centerbias = np.load('meta/centerbias.npy')
    centerbias -= logsumexp(centerbias)
    centerbias_data = centerbias[np.newaxis, :, :, np.newaxis]  # BHWC, 1 channel (log density)

    
    with tf.Graph().as_default():
        convT_filter = tf.convert_to_tensor(convT_filter_np, dtype=tf.float32)
        mean_t = tf.convert_to_tensor(mean_np, dtype=tf.float32)
        global_step = tf.Variable(0, trainable=False) 
        
        # input op
        cb_op = tf.placeholder(dtype=tf.float32, shape=[args.batch_size,args.h,args.w,1])
        img_op = tf.placeholder(dtype=tf.float32, shape=[args.batch_size,args.h,args.w,3])
        sal_op = tf.placeholder(dtype=tf.float32, shape=[args.batch_size,args.h,args.w,1])
        tf.summary.image('img', img_op)
        tf.summary.image('sal', sal_op)

        # pre-proc
        img_do_op =  tf.nn.avg_pool(img_op, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        img_do_op = img_do_op - mean_t
        
        # vgg features
        feat_op_d = model.feat_vgg19(img_do_op, is_training)
        
        # upscale features
        feat_op_d = model.feat_resize(feat_op_d, convT_filter)
        
        # readout nets
        readout_op_d = {}
        for i in range(readout_net_num):
            readout_op_d[i] = model.readout_net(feat_op_d, 'readout_network_%d'%i)

        # centerbias combination
        cb_do_op = tf.nn.avg_pool(cb_op, ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME')
        wo_cb_op_d, w_cb_op_d = {},{} # with centerbias (cb), without (wo) centerbias
        for i in range(readout_net_num):
            wo_cb_op_d[i], w_cb_op_d[i] = model.combine_centerbias(readout_op_d[i], 
                    cb_do_op, 'saliency_map_%d'%i)

        # upscale log density
        for i in range(readout_net_num):
            wo_cb_op_d[i] = model.upscale_prob(wo_cb_op_d[i], 'upscale_log_density_%d'%(2*i+1))
            if ADD_CENTERBIAS:
                w_cb_op_d[i] = model.upscale_prob(w_cb_op_d[i], 'upscale_log_density_%d'%(2*i))
        print('Upscale OK. Now concat em all\n') # (Run OK)

        # concat 'em all
        wo_cb_op = tf.concat(list(wo_cb_op_d.values()), axis=3)
        wo_cb_op = tf_logsumexp(wo_cb_op, axis=3)
        wo_cb_op = tf.expand_dims(wo_cb_op, axis=3)
        wo_cb_op = tf.reduce_logsumexp(wo_cb_op, axis=[1,2])

        if ADD_CENTERBIAS:
            w_cb_op = tf.concat(list(w_cb_op_d.values()), axis=3)
            w_cb_op = tf_logsumexp(w_cb_op, axis=3)
            w_cb_op = tf.expand_dims(w_cb_op, axis=3)
        
        # loss and train
        loss_op = loss.l2_loss(wo_cb_op, sal_op, args) 
        train_op = loss.train(loss_op, global_step, args) 

        summary_op = tf.summary.merge_all()
        with tf.Session() as sess:
            print('Restoring net ...')
            restore_start_time = time.time()
            sess.run(tf.global_variables_initializer())
            if args.start == 1:
                weights_fn = 'meta/deep_gaze.npy'
                global_step = 0
                init_weights(sess, weights_fn)
            else:
                ckpt = tf.train.get_checkpoint_state(train_log_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    # Restores from checkpoint
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    print("Restore done. Step: ", global_step)
                else:
                    print('Error in network init. Abort.')
                    return
            restore_duration = time.time() - restore_start_time
            print('Network restored: %d:%02d'%(int(restore_duration/60), restore_duration%60))

            # Start the queue runners.
            coord = tf.train.Coordinator()
            try:
                threads = []
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                  threads.extend(qr.create_threads(sess, coord=coord, daemon=True,start=True))
                summary_writer = tf.summary.FileWriter(train_log_dir, graph_def=sess.graph_def)
                
                current_epoch = 0
                step = int(global_step)

                while current_epoch < args.epochs:
                    print('step: %d'%step)
                    next_epoch, img_batch, sal_batch = dataloader.next_batch()
                    start_time = time.time()
                    
                    #wo_cb = sess.run(wo_cb_op,
                    #    feed_dict={img_op: img_batch, sal_op: sal_batch})
                    #plt.figure(1)
                    #n, bins, patches = plt.hist(np.reshape(wo_cb, -1), bins='auto') 
                    #plt.savefig('hist_grad.png')
                    #plt.close()
                    #hist_grad = cv2.imread('hist_grad.png')
                    #cv2.imshow('hist_grad', hist_grad)
                    #cv2.waitKey(0)
                    #continue

                    _, loss_value = sess.run([train_op, loss_op],
                        feed_dict={img_op: img_batch, sal_op: sal_batch})
                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                    duration = time.time() - start_time
                    step+=1
                    
                    if step % args.display_interval == 0:
                        num_examples_per_step = args.batch_size
                        examples_per_sec = num_examples_per_step / duration
                        sec_per_batch = float(duration)
                        format_str = ('%s: epoch %d/%d, step %d, loss=%.3f (%.1f examples/sec; %.3f ' 'sec/batch)')
                        print (format_str % (datetime.now(), current_epoch, args.epochs, step, 
                            loss_value, examples_per_sec, sec_per_batch))
                        
                    if (step % args.summary_interval==0):
                        summary_str = sess.run(summary_op, 
                                feed_dict={img_op: img_batch, sal_op: sal_batch})
                        summary_writer.add_summary(summary_str, step)
                    
                    if next_epoch==1:
                        current_epoch +=1

                # Save model
                summary_str = sess.run(summary_op,
                        feed_dict={img_op: img_batch, sal_op: sal_batch})
                summary_writer.add_summary(summary_str, step)
                #checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                #saver.save(sess, checkpoint_path, global_step=step)

            except Exception as e:  # pylint: disable=broad-except
                coord.request_stop(e)
            
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)


if __name__ == '__main__':  
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_id', type=str)
    parser.add_argument('--dataset_dir', type=str, help='path to >0 and <0 list file')
    parser.add_argument('--w', type=int, help='image width')
    parser.add_argument('--h', type=int, help='image height')
    parser.add_argument('--resize_img', type=int, help='Set to 1 to resize img')
    
    parser.add_argument('--xp_name', type=str, help='xp name')
    parser.add_argument('--epochs', type=int, help='num of epochs to train')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-03, help='learning rate')
    parser.add_argument('--adam_b1', type=float, default=0.9, help='adam beta1')
    parser.add_argument('--adam_b2', type=float, default=0.999, help='adam beta2')
    parser.add_argument('--adam_eps', type=float, default=1e-08, help='adam epsilon')
    parser.add_argument('--moving_average_decay', type=float, default=0.9999, help='')
    
    parser.add_argument('--train_log_dir', type=str, help='path to log dir')
    parser.add_argument('--display_interval', type=int, default=10, help='')
    parser.add_argument('--summary_interval', type=int, default=10, help='')
    parser.add_argument('--save_interval', type=int, default=1000000, help='')
    parser.add_argument('--start', type=int, default=1, help='Set to 1 if first train')
    args = parser.parse_args()
    
    train_log_dir = os.path.join(args.train_log_dir, args.xp_name)
    if tf.gfile.Exists(train_log_dir):  
        tf.gfile.DeleteRecursively(train_log_dir)
    tf.gfile.MakeDirs(train_log_dir)

    train_network(args)


