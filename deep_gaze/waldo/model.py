
"""
- netmork model
- loss
- optimizer
- summary
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import sys
import tarfile


from six.moves import urllib
import tensorflow as tf
from math import sqrt
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import array_ops

FLAGS = tf.app.flags.FLAGS

@ops.RegisterGradient("MaxPoolGradWithArgmax")
def _MaxPoolGradGradWithArgmax(op, grad):
  print(len(op.outputs))
  print(len(op.inputs))
  print(op.name)
  return (array_ops.zeros(
      shape=array_ops.shape(op.inputs[0]),
      dtype=op.inputs[0].dtype), array_ops.zeros(
          shape=array_ops.shape(op.inputs[1]), dtype=op.inputs[1].dtype),
          gen_nn_ops._max_pool_grad_grad_with_argmax(
              op.inputs[0],
              grad,
              op.inputs[2],
              op.get_attr("ksize"),
              op.get_attr("strides"),
              padding=op.get_attr("padding")))
              
def feat_vgg19(images, is_training, reuse=False):
    """ Network model
    Args:
      images: Images returned from distorted_inputs() or inputs().
    Returns:
      Logits.
    """
    #print('inference::input', images.get_shape())
    print('\nVGG-19 features')

    feat_dict = {}
   
    with tf.variable_scope('features1', reuse=reuse) as scope: #1
        
        with tf.variable_scope('conv1', reuse=reuse) as scope: #1
            conv1_1 = tf.layers.conv2d(inputs=images, filters=64, kernel_size=(3,3), padding="same",
                    name='conv1_1', kernel_initializer=tf.contrib.layers.xavier_initializer())
            conv1_1 = tf.nn.relu(conv1_1)
            print('conv1_1', conv1_1.get_shape())
  
            conv1_2 = tf.layers.conv2d(inputs=conv1_1, filters=64, kernel_size=(3, 3), padding="same",
                    name='conv1_2', kernel_initializer=tf.contrib.layers.xavier_initializer())
            conv1_2 = tf.nn.relu(conv1_2)
            pool1 = tf.nn.max_pool(conv1_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool')
            print('pool1', pool1.get_shape())
  
        
        with tf.variable_scope('conv2', reuse=reuse) as scope:
            conv2_1 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=(3, 3), padding="same", 
                    name = 'conv2_1', kernel_initializer=tf.contrib.layers.xavier_initializer())
            conv2_1 = tf.nn.relu(conv2_1)
            print('conv2_1', conv2_1.get_shape())

            conv2_2 = tf.layers.conv2d(inputs=conv2_1, filters=128, kernel_size=(3, 3), padding="same",
                    name = 'conv2_2', kernel_initializer=tf.contrib.layers.xavier_initializer())
            conv2_2 = tf.nn.relu(conv2_2)
            pool2 = tf.nn.max_pool(conv2_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool')
            print('pool2', pool2.get_shape())
  
        
        
        # block 3
        with tf.variable_scope('conv3', reuse=reuse) as scope:
            conv3_1 = tf.layers.conv2d(inputs=pool2, filters=256, kernel_size=(3, 3), padding="same",
                    name = 'conv3_1', kernel_initializer=tf.contrib.layers.xavier_initializer())
            conv3_1 = tf.nn.relu(conv3_1)
            print('conv3_1', conv3_1.get_shape())

            conv3_2 = tf.layers.conv2d(inputs=conv3_1, filters=256, kernel_size=(3, 3), padding="same",
                    name = 'conv3_2', kernel_initializer=tf.contrib.layers.xavier_initializer())
            conv3_2 = tf.nn.relu(conv3_2)
            print('conv3_2', conv3_2.get_shape())

            conv3_3 = tf.layers.conv2d(inputs=conv3_2, filters=256, kernel_size=(3, 3), padding="same",
                    name = 'conv3_3', kernel_initializer=tf.contrib.layers.xavier_initializer())
            conv3_3 = tf.nn.relu(conv3_3)
            print('conv3_3', conv3_3.get_shape())

            conv3_4 = tf.layers.conv2d(inputs=conv3_3, filters=256, kernel_size=(3, 3), padding="same",
                    name = 'conv3_4', kernel_initializer=tf.contrib.layers.xavier_initializer())
            conv3_4 = tf.nn.relu(conv3_4)
            pool3 = tf.nn.max_pool(conv3_4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool')

        
        # block 4
        with tf.variable_scope('conv4', reuse=reuse) as scope:
            conv4_1 = tf.layers.conv2d(inputs=pool3, filters=512, kernel_size=(3, 3), padding="same",
                    name = 'conv4_1', kernel_initializer=tf.contrib.layers.xavier_initializer())
            conv4_1 = tf.nn.relu(conv4_1)
            print('conv4_1', conv4_1.get_shape())

            conv4_2 = tf.layers.conv2d(inputs=conv4_1, filters=512, kernel_size=(3, 3), padding="same", 
                    name = 'conv4_2', kernel_initializer=tf.contrib.layers.xavier_initializer())
            conv4_2 = tf.nn.relu(conv4_2)
            print('conv4_2', conv4_2.get_shape())

            conv4_3 = tf.layers.conv2d(inputs=conv4_2, filters=512, kernel_size=(3, 3), padding="same",
                    name = 'conv4_3', kernel_initializer=tf.contrib.layers.xavier_initializer())
            conv4_3 = tf.nn.relu(conv4_3)
            print('conv4_3', conv4_3.get_shape())

            conv4_4 = tf.layers.conv2d(inputs=conv4_3, filters=512, kernel_size=(3, 3), padding="same",
                    name = 'conv4_4', kernel_initializer=tf.contrib.layers.xavier_initializer())
            conv4_4 = tf.nn.relu(conv4_4)
            pool4 = tf.nn.max_pool(conv4_4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool')
            print('pool4', pool4.get_shape())
        
        # block 5
        with tf.variable_scope('conv5', reuse=reuse) as scope:
            conv5_1 = tf.layers.conv2d(inputs=pool4, filters=512, kernel_size=(3, 3), padding="same",
                    name = 'conv5_1', kernel_initializer=tf.contrib.layers.xavier_initializer())
            relu5_1 = tf.nn.relu(conv5_1)
            print('conv5_1', relu5_1.get_shape())
            
            conv5_2 = tf.layers.conv2d(inputs=relu5_1, filters=512, kernel_size=(3, 3), padding="same",
                    name = 'conv5_2', kernel_initializer=tf.contrib.layers.xavier_initializer())
            conv5_2 = tf.nn.relu(conv5_2)
            print('conv5_2', conv5_2.get_shape())

            conv5_3 = tf.layers.conv2d(inputs=conv5_2, filters=512, kernel_size=(3, 3), padding="same",
                    name = 'conv5_3', kernel_initializer=tf.contrib.layers.xavier_initializer())
            relu5_3 = tf.nn.relu(conv5_3)
            print('conv5_3', relu5_3.get_shape())
 
            conv5_4 = tf.layers.conv2d(inputs=relu5_3, filters=512, kernel_size=(3, 3), padding="same",
                    name = 'conv5_4', kernel_initializer=tf.contrib.layers.xavier_initializer())
            conv5_4 = tf.nn.relu(conv5_4)
            pool5 = tf.nn.max_pool(conv5_4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool')
            print('pool5', pool5.get_shape())
        
        feat_dict['conv5_1'] = conv5_1 
        feat_dict['relu5_1'] = relu5_1 
        feat_dict['relu5_2'] = conv5_2 # actually relu5_2
        feat_dict['conv5_3'] = conv5_3
        feat_dict['relu5_4'] = conv5_4 # actually relu5_4
        feat_dict['pool5'] = pool5
        return feat_dict

def feat_resize(feat_op_d, conv_filter, reuse=False):
    print('\nUpscale vgg features')
   
    # TODO: try to find why the hell they need to downscale the img again
    # or was it to get the upscaling factor that I shamelessly read from the
    # graph and hardcoded ?
    #img_do2 =  tf.nn.avg_pool(img_do, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool')
    #img_do2_shape = img_do2.get_shape()
    up_factor = 8
    
    with tf.variable_scope('resize_feature', reuse=reuse) as scope:
        with tf.variable_scope('upscale', reuse=reuse) as scope:
            old_shape = feat_op_d['conv5_1'].get_shape()
            new_shape = [old_shape[0],  tf.Dimension(tf.Dimension(8)*old_shape[1]),
                    tf.Dimension(tf.Dimension(8)*old_shape[2]), old_shape[3]]
            #print(new_shape)
            new_shape = tf.stack(new_shape)
            feat_op_d['conv5_1'] = tf.nn.conv2d_transpose(feat_op_d['conv5_1'],
                    conv_filter, new_shape, [1,up_factor,up_factor,1])
            print('up-conv5_1.shape', feat_op_d['conv5_1'].get_shape())
            
    for i, key in enumerate(['relu5_1', 'relu5_2', 'conv5_3', 'relu5_4']):
        #print(key)
        with tf.variable_scope('resize_feature_%d'%i, reuse=reuse) as scope:
            with tf.variable_scope('upscale', reuse=reuse) as scope:
                old_shape = feat_op_d[key].get_shape()
                new_shape = [old_shape[0],  tf.Dimension(tf.Dimension(8)*old_shape[1]),
                    tf.Dimension(tf.Dimension(8)*old_shape[2]), old_shape[3]]
                new_shape = tf.stack(new_shape)
                feat_op_d[key] = tf.nn.conv2d_transpose(feat_op_d[key],
                        conv_filter, new_shape, [1,up_factor,up_factor,1])
                print('up-%s.shape'%key, feat_op_d[key].get_shape())

    return feat_op_d


def readout_net(feat_op_d, scope_str, reuse=False):

    if scope_str=='readout_network0':
        print('\nReadout nets')

    k = (1,1) # kernel size
            
    with tf.variable_scope(scope_str, reuse=reuse) as scope:

        conv1_part0 = tf.layers.conv2d(inputs=feat_op_d['conv5_1'], filters=16, kernel_size=k,
                padding="same", use_bias=False, name='conv1_part0')
        if scope_str=='readout_network0':
            print('%s: conv1_part0.shape: '%scope_str, conv1_part0.get_shape())

        conv1_part1 = tf.layers.conv2d(inputs=feat_op_d['relu5_1'], filters=16, kernel_size=k,
                padding="same", use_bias=False, name='conv1_part1')
        if scope_str=='readout_network0':
            print('%s: conv1_part1.shape: '%scope_str, conv1_part1.get_shape())

        conv1_part2 = tf.layers.conv2d(inputs=feat_op_d['relu5_2'], filters=16, kernel_size=k,
                padding="same", use_bias=False, name='conv1_part2')
        if scope_str=='readout_network0':
            print('%s: conv1_part2.shape: '%scope_str, conv1_part2.get_shape())

        conv1_part3 = tf.layers.conv2d(inputs=feat_op_d['conv5_3'], filters=16, kernel_size=k,
                padding="same", use_bias=False, name='conv1_part3')
        if scope_str=='readout_network0':
            print('%s: conv1_part3.shape: '%scope_str, conv1_part3.get_shape())

        conv1_part4 = tf.layers.conv2d(inputs=feat_op_d['relu5_4'], filters=16, kernel_size=k,
                padding="same", use_bias=False, name='conv1_part4')
        if scope_str=='readout_network0':
            print('%s: conv1_part4.shape: '%scope_str, conv1_part4.get_shape())
        
        with tf.variable_scope('conv1', reuse=reuse) as scope:
            biases = tf.get_variable('bias', [16], tf.float32, tf.constant_initializer(0.0))
            conv1 = conv1_part0 + conv1_part1 + conv1_part2 + conv1_part3 + conv1_part4
            conv1 = tf.nn.bias_add(conv1, biases) 
            conv1 = tf.nn.relu(conv1)


        conv2 = tf.layers.conv2d(inputs=conv1, filters=32, kernel_size=k, padding="same", name='conv2')
        conv2 = tf.nn.relu(conv2)
        if scope_str=='readout_network0':
            print('%s: conv2.shape: '%scope_str, conv2.get_shape())

        conv3 = tf.layers.conv2d(inputs=conv2, filters=2, kernel_size=k, padding="same", name='conv3')
        conv3 = tf.nn.relu(conv3)
        if scope_str=='readout_network0':
            print('%s: conv3.shape: '%scope_str, conv3.get_shape())

        conv4 = tf.layers.conv2d(inputs=conv3, filters=1, kernel_size=k,
                padding="same", use_bias=False, name='conv4')
        if scope_str=='readout_network0':
            print('%s: conv4.shape: '%scope_str, conv4.get_shape())

    return conv4


def get_gauss_1d(sigma, shape):
    mean = 20
    x = tf.cast(tf.range(41), tf.float32)
    k = tf.exp( - tf.pow(x-mean, 2) / (2*tf.pow(sigma,2)) )
    k_normed = k / tf.reduce_sum(k)
    k_reshaped = tf.reshape(k_normed, shape)
    return k_reshaped


def combine_centerbias(readout, centerbias, scope_str, reuse=False):
    display = (scope_str=='saliency_map_0')
    if display:
        print('\nCombine centerbias')
        feat_dict = {}
    with tf.variable_scope(scope_str, reuse=reuse) as scope: # saliency_map_%d
        alpha = tf.get_variable('alpha', [1], tf.float32, tf.constant_initializer(0.0))
        c = alpha*centerbias

        sigma = tf.get_variable('sigma', [1], tf.float32, tf.constant_initializer(0.0))

        if display:
            print('readout.shape', readout.get_shape())

        with tf.name_scope('blur'):
            with tf.name_scope('gauss_1d'):
                gaussian_kernel = get_gauss_1d(sigma, [-1,1,1,1])
                if display:
                    print('\ngauss_1d::gaussian_kernel.shape', gaussian_kernel.get_shape())

                with tf.name_scope('replication_padding'):
                    a = tf.strided_slice(readout, [0,0], [0,1], [1,1], begin_mask=1, end_mask=1)
                    a = tf.tile(a, [ 1,20,1,1])
                    if display:
                        feat_dict['a'] = a
                        print('gauss_1d::a.shape', a.get_shape())

                    b = tf.strided_slice(readout, [0,-1], [0,0], [1, 1], begin_mask=1, end_mask=3)
                    b = tf.tile(b, [ 1,20,1 ,1])
                    if display:
                        feat_dict['b'] = b
                        print('gauss_1d::b.shape', b.get_shape())

                    padded = tf.concat([a, readout, b], axis=1)

                    if display:
                        feat_dict['padded'] = padded
                        print('gauss_1d::padded.shape', padded.get_shape())
                blur1 = tf.nn.conv2d(padded, gaussian_kernel, [1,1,1,1], padding="VALID", name='gaussian_convolution')
                if display:
                    print('gauss_1d::blur1.shape: ', blur1.get_shape())

            with tf.name_scope('gauss_1d_1'):
                gaussian_kernel = get_gauss_1d(sigma, [ 1,-1, 1, 1])
                if display:
                    print('\ngauss_1d_1::gaussian_kernel.shape', gaussian_kernel.get_shape())

                with tf.name_scope('replication_padding'):
                    a = tf.strided_slice(blur1, [0,0,0], [0,0,1], [1,1,1], begin_mask=3, end_mask=3)
                    a = tf.tile(a, [ 1, 1, 20, 1])
                    if display:
                        print('gauss_1d_1::a.shape', a.get_shape())

                    b = tf.strided_slice(blur1, [0,0,-1], [0,0,0], [1,1,1], begin_mask=3, end_mask=7)
                    b = tf.tile(b, [ 1, 1, 20,  1])
                    if display:
                        print('gauss_1d_1::b.shape', b.get_shape())

                    padded = tf.concat([a, blur1, b], axis=2)
                    if display:
                        print('gauss_1d_1::padded.shape', padded.get_shape())
                blur2 = tf.nn.conv2d(padded, gaussian_kernel, [1,1,1,1], padding="VALID", name='gaussian_convolution')
                if display:
                    print('gauss_1d::blur1.shape: ', blur1.get_shape())

        # wo centerbias
        # two logsumexp because they process one axis at a time
        # (all this block computes a softmax but I don't know why they chose to
        # compute it this way, can probably be better written unless I am
        # missing something)
        log_sum_exp_wo = tf.reduce_logsumexp(blur2, axis=1)
        log_sum_exp_wo = tf.reduce_logsumexp(log_sum_exp_wo, axis=1)
        log_sum_exp_wo = tf.expand_dims(tf.expand_dims(log_sum_exp_wo, 1), 1)
        saliency_map_prob_wo = blur2 - log_sum_exp_wo
        if display:
            print('saliency_map_prob_wo.shape: ', saliency_map_prob_wo.get_shape())

        # w centerbias
        with_centerbias = alpha * centerbias + blur2
        log_sum_exp = tf.reduce_logsumexp(with_centerbias, axis=1)
        log_sum_exp = tf.reduce_logsumexp(log_sum_exp, axis=1)
        log_sum_exp = tf.expand_dims(tf.expand_dims(log_sum_exp, 1), 1)
        saliency_map_prob_w = with_centerbias - log_sum_exp 
        if display:
            print('saliency_map_prob_w.shape: ', saliency_map_prob_w.get_shape())
        
        #return feat_dict
        return saliency_map_prob_wo, saliency_map_prob_w



def upscale_prob(saliency_map_prob, scope_str, reuse=False):
    """
    Args:
        shape: img shape
    """
    display = (scope_str=='upscale_log_density_1')
    if display:
        print('\nUpscale log density')

    conv_filter = tf.ones([4,4,1,1])
    up_factor = 4

    with tf.variable_scope(scope_str, reuse=reuse) as scope:
        with tf.name_scope('upscale'):
            old_shape = saliency_map_prob.get_shape()
            new_shape = [old_shape[0],  tf.Dimension(tf.Dimension(up_factor)*old_shape[1]),
                    tf.Dimension(tf.Dimension(up_factor)*old_shape[2]), old_shape[3]]
            #print(new_shape)
            new_shape = tf.stack(new_shape)
            upscaled = tf.nn.conv2d_transpose(saliency_map_prob,
                    conv_filter, new_shape, [1,up_factor,up_factor,1])
            if display:
                print('%s::upscaled.shape'%scope_str, upscaled.get_shape())

        with tf.name_scope('normalize_log_density'):
            log_sum_exp = tf.reduce_logsumexp(upscaled, axis=1)
            log_sum_exp = tf.reduce_logsumexp(log_sum_exp, axis=1)
            log_sum_exp = tf.expand_dims(tf.expand_dims(log_sum_exp, 1), 1)
            upscaled = upscaled - log_sum_exp

    return upscaled












def features1_bn(images, is_training, reuse=False):
    """ Network model with optinal bn. You need to activate it.
    Args:
      images: Images returned from distorted_inputs() or inputs().
    Returns:
      Logits.
    """
    print('inference::input', images.get_shape())
    bn = False
    feat, grads_dict = {},{}
   
    with tf.variable_scope('features1', reuse=reuse) as scope: #1
        
        with tf.variable_scope('conv1', reuse=reuse) as scope: #1
            with tf.variable_scope('conv1_1', reuse=reuse) as scope: #1
                conv1_1 = tf.layers.conv2d(inputs=images, filters=64, kernel_size=(3,3),
                        padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
                conv1_1 = tf.nn.relu(conv1_1)
                print('conv1_1', conv1_1.get_shape())
  
            with tf.variable_scope('conv1_2', reuse=reuse) as scope: #2
                conv1_2 = tf.layers.conv2d(inputs=conv1_1, filters=64, kernel_size=(3, 3),
                        padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
                if bn:
                    conv1_2 = tf.contrib.layers.batch_norm(conv1_2, fused=True, decay=0.9, is_training=is_training)
                conv1_2 = tf.nn.relu(conv1_2)
                pool1 = tf.nn.max_pool(conv1_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool')
                print('pool1', pool1.get_shape())
  
        
        with tf.variable_scope('conv2', reuse=reuse) as scope:
            with tf.variable_scope('conv2_1', reuse=reuse) as scope:#3
                conv2_1 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=(3, 3),
                        padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
                if bn:
                    conv2_1 = tf.contrib.layers.batch_norm(conv2_1, fused=True, decay=0.9, is_training=is_training)
                conv2_1 = tf.nn.relu(conv2_1)
                print('conv2_1', conv2_1.get_shape())

            with tf.variable_scope('conv2_2', reuse=reuse) as scope:#4
                conv2_2 = tf.layers.conv2d(inputs=conv2_1, filters=128, kernel_size=(3, 3),
                    padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
                if bn:
                    conv2_2 = tf.contrib.layers.batch_norm(conv2_2, fused=True, decay=0.9, is_training=is_training)
                conv2_2 = tf.nn.relu(conv2_2)
                pool2 = tf.nn.max_pool(conv2_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool') #pool2
                #feat['pool2'] = pool2
                print('pool2', pool2.get_shape())
  
        
        
        with tf.variable_scope('conv3', reuse=reuse) as scope:
            with tf.variable_scope('conv3_1', reuse=reuse) as scope:#5
                conv3_1 = tf.layers.conv2d(inputs=pool2, filters=256, kernel_size=(3, 3),
                        padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
                if bn:
                    conv3_1 = tf.contrib.layers.batch_norm(conv3_1, fused=True, decay=0.9, is_training=is_training)
                conv3_1 = tf.nn.relu(conv3_1)
                print('conv3_1', conv3_1.get_shape())

            with tf.variable_scope('conv3_2', reuse=reuse) as scope:#6
                conv3_2 = tf.layers.conv2d(inputs=conv3_1, filters=256, kernel_size=(3, 3),
                        padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
                if bn:
                    conv3_2 = tf.contrib.layers.batch_norm(conv3_2, fused=True, decay=0.9, is_training=is_training)
                conv3_2 = tf.nn.relu(conv3_2)
                print('conv3_2', conv3_2.get_shape())

            with tf.variable_scope('conv3_3', reuse=reuse) as scope:#7
                conv3_3 = tf.layers.conv2d(inputs=conv3_2, filters=256, kernel_size=(3, 3),
                        padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
                if bn:
                    conv3_3 = tf.contrib.layers.batch_norm(conv3_3, fused=True, decay=0.9, is_training=is_training)
                conv3_3 = tf.nn.relu(conv3_3)

            with tf.variable_scope('conv3_4', reuse=reuse) as scope:#7
                conv3_3 = tf.layers.conv2d(inputs=conv3_3, filters=256, kernel_size=(3, 3),
                        padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
                conv3_4 = tf.nn.relu(conv3_4)
                pool3 = tf.nn.max_pool(conv3_4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool') #pool3

        
        with tf.variable_scope('conv4', reuse=reuse) as scope:
            with tf.variable_scope('conv4_1', reuse=reuse) as scope:# 8
                conv4_1 = tf.layers.conv2d(inputs=pool3, filters=512, kernel_size=(3, 3),
                        padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
                if bn:
                    conv4_1 = tf.contrib.layers.batch_norm(conv4_1, fused=True, decay=0.9, is_training=is_training)
                conv4_1 = tf.nn.relu(conv4_1)
                print('conv4_1', conv4_1.get_shape())

            with tf.variable_scope('conv4_2', reuse=reuse) as scope:#9
                conv4_2 = tf.layers.conv2d(inputs=conv4_1, filters=512, kernel_size=(3, 3),
                        padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
                if bn:
                    conv4_2 = tf.contrib.layers.batch_norm(conv4_2, fused=True, decay=0.9, is_training=is_training)
                conv4_2 = tf.nn.relu(conv4_2)
                print('conv4_2', conv4_2.get_shape())

            with tf.variable_scope('conv4_3', reuse=reuse) as scope:#10
                conv4_3 = tf.layers.conv2d(inputs=conv4_2, filters=512, kernel_size=(3, 3),
                        padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
                if bn:
                    conv4_3 = tf.contrib.layers.batch_norm(conv4_3, fused=True, decay=0.9, is_training=is_training)
                conv4_3 = tf.nn.relu(conv4_3)

            with tf.variable_scope('conv4_3', reuse=reuse) as scope:#10
                conv4_3 = tf.layers.conv2d(inputs=conv4_2, filters=512, kernel_size=(3, 3),
                        padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
                if bn:
                    conv4_3 = tf.contrib.layers.batch_norm(conv4_3, fused=True, decay=0.9, is_training=is_training)
                conv4_3 = tf.nn.relu(conv4_3)
                pool4 = tf.nn.max_pool(conv4_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool')
                print('pool4', pool4.get_shape())


        with tf.variable_scope('conv4', reuse=reuse) as scope:
            with tf.variable_scope('conv5_1', reuse=reuse) as scope:#11
                conv5_1 = tf.layers.conv2d(inputs=pool4, filters=512, kernel_size=(3, 3),
                        padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
                if bn:
                    conv5_1 = tf.contrib.layers.batch_norm(conv5_1, fused=True, decay=0.9, is_training=is_training)
                conv5_1 = tf.nn.relu(conv5_1)
                print('conv5_1', conv5_1.get_shape())
            
            with tf.variable_scope('conv5_2', reuse=reuse) as scope:#12
                conv5_2 = tf.layers.conv2d(inputs=conv5_1, filters=512, kernel_size=(3, 3),
                        padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
                if bn:
                    conv5_2 = tf.contrib.layers.batch_norm(conv5_2, fused=True, decay=0.9, is_training=is_training)
                conv5_2 = tf.nn.relu(conv5_2)
                print('conv5_2', conv5_2.get_shape())

            with tf.variable_scope('conv5_3', reuse=reuse) as scope:#13
                conv5_3 = tf.layers.conv2d(inputs=conv5_2, filters=512, kernel_size=(3, 3),
                        padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
                if bn:
                    conv5_3 = tf.contrib.layers.batch_norm(conv5_3, fused=True, decay=0.9, is_training=is_training)
                conv5_3 = tf.nn.relu(conv5_3)
                pool5 = tf.nn.max_pool(conv5_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool') #pool5
                #pool5 = tf.nn.l2_normalize(pool5, dim=3, epsilon=1e-12)
                print('pool5', pool5.get_shape())
 
            with tf.variable_scope('conv5_4', reuse=reuse) as scope:#13
                conv5_4 = tf.layers.conv2d(inputs=conv5_2, filters=512, kernel_size=(3, 3),
                        padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
                if bn:
                    conv5_4 = tf.contrib.layers.batch_norm(conv5_4, fused=True, decay=0.9, is_training=is_training)
                conv5_4 = tf.nn.relu(conv5_4)
                pool5 = tf.nn.max_pool(conv5_4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool') #pool5
                #pool5 = tf.nn.l2_normalize(pool5, dim=3, epsilon=1e-12)
                print('pool5', pool5.get_shape())
     
        return pool5


