"""
Shamelessly copy-pasted from https://github.com/uzh-rpg/netvlad_tf_open
"""
import tensorflow as tf

def netVLAD(inputs, num_clusters, assign_weight_initializer=None, 
            cluster_initializer=None, skip_postnorm=False, reuse=False):
    ''' skip_postnorm: Only there for compatibility with mat files. '''

    # TODO: normalisation
    with tf.variable_scope('netvlad', reuse=reuse) as scope:

        K = num_clusters
        # D: number of (descriptor) dimensions.
        D = inputs.get_shape()[-1]

        # soft-assignment.
        s = tf.layers.conv2d(inputs, filters=K, kernel_size=1, use_bias=False,
                kernel_initializer=tf.contrib.layers.xavier_initializer(), 
                name='assignment')
        #print('s.shape: ', s.get_shape()) # (bz, H, W, K)


        a = tf.nn.softmax(s)
        #print('a.shape: ', a.get_shape()) # (bz, H, W, K)

        # Dims used hereafter: batch, H, W, desc_coeff, cluster
        # Move cluster assignment to corresponding dimension.
        a = tf.expand_dims(a, -2)

        # VLAD core.
        C = tf.get_variable('cluster_centers', [1, 1, 1, D, K],
                            initializer=cluster_initializer,
                            dtype=inputs.dtype)
        #print('C.shape: ', C.get_shape())

        #print('tf.expand_dims(inputs, -1).shape: ', tf.expand_dims(inputs, -1).get_shape())
        v = tf.expand_dims(inputs, -1) + C
        #print('v.shape: ', v.get_shape())
        v = a * v
        #print('a * v.shape: ', v.get_shape())
        v = tf.reduce_sum(v, axis=[1, 2])
        #print('tf.reduce_sum(a * v, axis=[1,2]).shape: ', v.get_shape())
        v = tf.transpose(v, perm=[0, 2, 1])
        #print('tf.transpose(v, perm=[0,2,1]): ', v.get_shape())

        # TODO: intra normalisation
        # TODO: L2 normalisation

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
