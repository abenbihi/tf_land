import numpy as np
import os
from os.path import dirname
import tensorflow as tf

def defaultCheckpoint():
    return os.path.join(dirname(dirname(dirname(__file__))), 
                              'checkpoints', 
                              'vd16_pitts30k_conv5_3_vlad_preL2_intra_white')

def netVLAD(inputs, num_clusters, assign_weight_initializer=None, 
            cluster_initializer=None, skip_postnorm=False):
    ''' skip_postnorm: Only there for compatibility with mat files. '''
    K = num_clusters
    # D: number of (descriptor) dimensions.
    D = inputs.get_shape()[-1]

    # soft-assignment.
    s = tf.layers.conv2d(inputs, K, 1, use_bias=False,
                         kernel_initializer=assign_weight_initializer,
                         name='assignment')
    a = tf.nn.softmax(s)

    # Dims used hereafter: batch, H, W, desc_coeff, cluster
    # Move cluster assignment to corresponding dimension.
    a = tf.expand_dims(a, -2)

    # VLAD core.
    C = tf.get_variable('cluster_centers', [1, 1, 1, D, K],
                        initializer=cluster_initializer,
                        dtype=inputs.dtype)

    v = tf.expand_dims(inputs, -1) + C
    v = a * v
    v = tf.reduce_sum(v, axis=[1, 2])
    v = tf.transpose(v, perm=[0, 2, 1])

    if not skip_postnorm:
        # Result seems to be very sensitive to the normalization method
        # details, so sticking to matconvnet-style normalization here.
        v = matconvnetNormalize(v, 1e-12)
        v = tf.transpose(v, perm=[0, 2, 1])
        v = matconvnetNormalize(tf.layers.flatten(v), 1e-12)

    return v


def matconvnetNormalize(inputs, epsilon):
    return inputs / tf.sqrt(tf.reduce_sum(inputs ** 2, axis=-1, keep_dims=True)
                            + epsilon)

def vgg16NetvladPca(image_batch, reuse=False):
    ''' Assumes rank 4 input, first 3 dims fixed or dynamic, last dim 1 or 3. 
    '''
    assert len(image_batch.shape) == 4
    
    with tf.variable_scope('vgg16_netvlad_pca', reuse=reuse):
        # Gray to color if necessary.
        if image_batch.shape[3] == 1:
            x = tf.nn.conv2d(image_batch, np.ones((1, 1, 1, 3)), 
                                  np.ones(4).tolist(), 'VALID')
        else :
            assert image_batch.shape[3] == 3
            x = image_batch
        
        ## Subtract trained average image.
        #average_rgb = tf.get_variable(
        #        'average_rgb', 3, dtype=image_batch.dtype)
        #x = x - average_rgb
        
        # VGG16
        def vggConv(inputs, numbers, out_dim, with_relu):
            if with_relu:
                activation = tf.nn.relu
            else:
                activation = None
            return tf.layers.conv2d(inputs, out_dim, [3, 3], 1, padding='same',
                                    activation=activation, 
                                    name='conv%s' % numbers)
        def vggPool(inputs):
            return tf.layers.max_pooling2d(inputs, 2, 2)
        
        x = vggConv(x, '1_1', 64, True)
        x = vggConv(x, '1_2', 64, False)
        x = vggPool(x)
        x = tf.nn.relu(x)
        
        x = vggConv(x, '2_1', 128, True)
        x = vggConv(x, '2_2', 128, False)
        x = vggPool(x)
        x = tf.nn.relu(x)
        
        x = vggConv(x, '3_1', 256, True)
        x = vggConv(x, '3_2', 256, True)
        x = vggConv(x, '3_3', 256, False)
        x = vggPool(x)
        x = tf.nn.relu(x)
        
        x = vggConv(x, '4_1', 512, True)
        x = vggConv(x, '4_2', 512, True)
        x = vggConv(x, '4_3', 512, False)
        x = vggPool(x)
        x = tf.nn.relu(x)
        
        x = vggConv(x, '5_1', 512, True)
        x = vggConv(x, '5_2', 512, True)
        x = vggConv(x, '5_3', 512, False)
        
        # NetVLAD
        x = tf.nn.l2_normalize(x, dim=-1)
        x = netVLAD(x, 64)
        
        # PCA
        x = tf.layers.conv2d(tf.expand_dims(tf.expand_dims(x, 1), 1), 
                             4096, 1, 1, name='WPCA')
        x = tf.nn.l2_normalize(tf.layers.flatten(x), dim=-1)
        
    return x
