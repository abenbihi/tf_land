# We instantiate all variables using tf.get_variable() instead of
# tf.Variable() in order to share variables across multiple GPU training runs.
# If we only ran this model on a single GPU, we could simplify this function
# by replacing all instances of tf.get_variable() with tf.Variable().

"""Builds the CIFAR-10 network.

Summary of available functions:

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
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
              
#data_format='NHWC'))

def model(images):
    """Build the CIFAR-10 model.
    Args:
      images: Images returned from distorted_inputs() or inputs().
    Returns:
      Logits.
    """
    
    size_filter=3
    i=1
    layers=[]
    conv_=[]
    arg_=[]
    print('inference::input', images.get_shape())
 
    with tf.variable_scope('conv1_1') as scope: #1
        conv1_1 = tf.layers.conv2d(inputs=images, filters=64, kernel_size=(3,3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv1_1 = tf.nn.relu(conv1_1)
        print('conv1_1', conv1_1.get_shape())
  
    with tf.variable_scope('conv1_2') as scope: #2
        conv1_2 = tf.layers.conv2d(inputs=conv1_1, filters=64, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv1_2 = tf.nn.relu(conv1_2)
        pool1 = tf.nn.max_pool(conv1_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool') #pool1
        print('pool1', pool1.get_shape())
  
    with tf.variable_scope('conv2_1') as scope:#3
        conv2_1 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv2_1 = tf.nn.relu(conv2_1)
        print('conv2_1', conv2_1.get_shape())

    with tf.variable_scope('conv2_2') as scope:#4
        conv2_2 = tf.layers.conv2d(inputs=conv2_1, filters=128, kernel_size=(3, 3),
            padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv2_2 = tf.nn.relu(conv2_2)
        pool2 = tf.nn.max_pool(conv2_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool2
        print('pool2', pool2.get_shape())
  
    with tf.variable_scope('conv3_1') as scope:#5
        conv3_1 = tf.layers.conv2d(inputs=pool2, filters=256, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv3_1 = tf.nn.relu(conv3_1)
        print('conv3_1', conv3_1.get_shape())

    with tf.variable_scope('conv3_2') as scope:#6
        conv3_2 = tf.layers.conv2d(inputs=conv3_1, filters=256, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv3_2 = tf.nn.relu(conv3_2)
        print('conv3_2', conv3_2.get_shape())

    with tf.variable_scope('conv3_3') as scope:#7
        conv3_3 = tf.layers.conv2d(inputs=conv3_2, filters=256, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv3_3 = tf.nn.relu(conv3_3)
        pool3 = tf.nn.max_pool(conv3_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool3
        print('pool3', pool3.get_shape())

    with tf.variable_scope('conv4_1') as scope:# 8
        conv4_1 = tf.layers.conv2d(inputs=pool3, filters=512, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv4_1 = tf.nn.relu(conv4_1)
        print('conv4_1', conv4_1.get_shape())

    with tf.variable_scope('conv4_2') as scope:#9
        conv4_2 = tf.layers.conv2d(inputs=conv4_1, filters=512, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv4_2 = tf.nn.relu(conv4_2)
        print('conv4_2', conv4_2.get_shape())

    with tf.variable_scope('conv4_3') as scope:#10
        conv4_3 = tf.layers.conv2d(inputs=conv4_2, filters=512, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv4_3 = tf.nn.relu(conv4_3)
        pool4 = tf.nn.max_pool(conv4_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool4
        print('pool4', pool4.get_shape())


    with tf.variable_scope('conv5_1') as scope:#11
        conv5_1 = tf.layers.conv2d(inputs=pool4, filters=512, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv5_1 = tf.nn.relu(conv5_1)
        print('conv5_1', conv5_1.get_shape())
    
    with tf.variable_scope('conv5_2') as scope:#12
        conv5_2 = tf.layers.conv2d(inputs=conv5_1, filters=512, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv5_2 = tf.nn.relu(conv5_2)
        print('conv5_2', conv5_2.get_shape())

    with tf.variable_scope('conv5_3') as scope:#13
        conv5_3 = tf.layers.conv2d(inputs=conv5_2, filters=512, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv5_3 = tf.nn.relu(conv5_3)
        print('conv5_3', conv5_3.get_shape())
    

    #### upsameple
    with tf.variable_scope('score-dsn1') as scope:
        score_dsn1_up = tf.layers.conv2d(inputs=conv1_2, filters=1, kernel_size=(1, 1),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        print('score_dsn1_up', score_dsn1_up.get_shape())
        #upscore_dsn1 = tf.crop(score_dsn1_up)
        #print('score_dsn1_up', score_dsn1_up.get_shape())
        upscore_dsn1 = score_dsn1_up
        #upscore_dsn1 = tf.sigmoid(upscore_dsn1)
 

    with tf.variable_scope('score-dsn2') as scope:
        score_dsn2 = tf.layers.conv2d(inputs=conv2_2, filters=1, kernel_size=(1, 1),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        print('score_dsn2::conv', score_dsn2.get_shape())
   
    with tf.variable_scope('upsample_2') as scope:
        score_dsn2_up = tf.layers.conv2d_transpose(score_dsn2,
                filters= 1, kernel_size=(4,4), strides=(2,2), padding='same',
                use_bias=False,
                kernel_initializer=tf.contrib.layers.xavier_initializer())
        print('score_dsn2::deconv', score_dsn2_up.get_shape())
        upscore_dsn2 = score_dsn2_up


    with tf.variable_scope('score-dsn3') as scope:
        score_dsn3 = tf.layers.conv2d(inputs=conv3_3, filters=1, kernel_size=(1, 1),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        print('score_dsn3::conv', score_dsn3.get_shape())
   
    with tf.variable_scope('upsample_4') as scope:
        score_dsn3_up = tf.layers.conv2d_transpose(score_dsn3,
                filters= 1, kernel_size=(8,8),strides=(4,4), padding='same', 
                use_bias=False,
                kernel_initializer=tf.contrib.layers.xavier_initializer())
        print('score_dsn3::deconv', score_dsn3_up.get_shape())
        upscore_dsn3 = score_dsn3_up


    with tf.variable_scope('score-dsn4') as scope:
        score_dsn4 = tf.layers.conv2d(inputs=conv4_3, filters=1, kernel_size=(1, 1),
               padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        print('score_dsn4::conv', score_dsn4.get_shape())
   
    with tf.variable_scope('upsample_8') as scope:
        score_dsn4_up = tf.layers.conv2d_transpose(score_dsn4,
               filters= 1, kernel_size=(16,16), strides=(8,8), padding='same', 
               use_bias=False,
               kernel_initializer=tf.contrib.layers.xavier_initializer())
        print('score_dsn4::deconv', score_dsn4_up.get_shape())
        upscore_dsn4 = score_dsn4_up


    with tf.variable_scope('score-dsn5') as scope:
        score_dsn5 = tf.layers.conv2d(inputs=conv5_3, filters=1, kernel_size=(1, 1),
              padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        print('score_dsn5::conv', score_dsn5.get_shape())
   
    with tf.variable_scope('upsample_16') as scope:
        score_dsn5_up = tf.layers.conv2d_transpose(score_dsn5,
              filters= 1, kernel_size=(32,32), strides=(16,16), padding='same', 
               use_bias=False,
              kernel_initializer=tf.contrib.layers.xavier_initializer())
        print('score_dsn5::deconv', score_dsn5_up.get_shape())
        upscore_dsn5 = score_dsn5_up


    with tf.variable_scope('new-score-weighting') as scope:
        concat_upscore = tf.concat([upscore_dsn1, upscore_dsn2, upscore_dsn3,
            upscore_dsn4, upscore_dsn5], 3)
        print('concat.shape', concat_upscore.get_shape())
        upscore_fuse = tf.layers.conv2d(inputs=concat_upscore, filters=1, kernel_size=(1, 1),
              padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        print('upscore_fuse.shape', upscore_fuse.get_shape())
        #upscore_fuse = tf.sigmoid(upscore_fuse)
       
    out = {}
    out['fuse'] = upscore_fuse
    out['dsn1'] = upscore_dsn1
    out['dsn2'] = upscore_dsn2
    out['dsn3'] = upscore_dsn3
    out['dsn4'] = upscore_dsn4
    out['dsn5'] = upscore_dsn5

    return out


def loss(logits, labels):
  """Add L2Loss to all the trainable variables.

  Add summary for for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 4-D tensor
            of shape [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1]

  Returns:
    Loss tensor of type float.
  """
  print('\n*** Loss ***')
  print('loss::preds.shape', logits.get_shape())
  print('loss::labels.shape', labels.get_shape())
  #loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels))
  #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels))

  loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
      logits=logits))
  tf.summary.scalar('loss', loss)
  tf.add_to_collection('losses', loss)
  # The total loss is defined as the l2 loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name +' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step, args):
  """Train CIFAR-10 model.
  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.
  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  print('\n*** Train ***')
  # Variables that affect learning rate.
  #num_batches_per_epoch = header.TRAIN_SET_SIZE / header.BATCH_SIZE

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    #opt = tf.train.GradientDescentOptimizer(lr)
    opt = tf.train.AdamOptimizer(args.lr, args.adam_b1, args.adam_b2,
            args.adam_eps)
    update_ops =  tf.get_collection(tf.GraphKeys.UPDATE_OPS) #line for BN
    with tf.control_dependencies(update_ops):
        grads = opt.compute_gradients(total_loss)
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(0.9999, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op


