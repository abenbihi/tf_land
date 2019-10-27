"""
IMplements the weird logsumexp of the last layer (post readout concat, when
someone used mean instead of sum in the logsumexp. Don't ask why, I don't know
either
"""
import numpy as np

# needed for the weird logsumexp
import tensorflow as tf
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.framework import common_shapes
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops

def fucking_deep_gaze_logsumexp(input_tensor,axis=None, keepdims=False,
        name=None):
    """
    Adaptd from
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/math_ops.py.
    It is the same as the classic logsumexp instead you substact log(N) where N
    in the number of tensor over which compute the logsumexp (if you have 10
    readout nets, N=10). I don't know why they do this.
    """
    keepdims = False if keepdims is None else keepdims
    input_tensor = ops.convert_to_tensor(input_tensor)
    with ops.name_scope(name, "ReduceLogSumExp", [input_tensor]) as name:
        raw_max = tf.reduce_max(input_tensor, axis=axis, keep_dims=True)
        my_max = array_ops.stop_gradient( array_ops.where(
            gen_math_ops.is_finite(raw_max), raw_max,
            array_ops.zeros_like(raw_max)))
        result = gen_math_ops.log(
                #reduce_sum( # normal logsumexp
                tf.reduce_mean( # fuckimg modif from deep_gaze for the output only
                    gen_math_ops.exp(tf.subtract(input_tensor, my_max)),
                    axis, keep_dims=keepdims))
        if not keepdims:
            my_max = array_ops.reshape(my_max, array_ops.shape(result))
        result = gen_math_ops.add(result, my_max)
        return result


def softmax_2d(input_tensor, axis=None, keepdims=False, name=None):
    """
    Adaptd from
    https://gist.github.com/raingo/a5808fe356b8da031837
    """
    keepdims = False if keepdims is None else keepdims
    input_tensor = ops.convert_to_tensor(input_tensor)
    with ops.name_scope(name, "softmax_2d", [input_tensor]) as name:
        raw_max = tf.reduce_max(input_tensor, axis=axis, keep_dims=True)
        my_max = array_ops.stop_gradient( array_ops.where(
            gen_math_ops.is_finite(raw_max), raw_max,
            array_ops.zeros_like(raw_max)))
        target_exp = gen_math_ops.exp(tf.subtract(input_tensor, my_max))
        normalize = tf.reduce_sum(target_exp, axis, keep_dims=True)
        softmax = target_exp / normalize
        return softmax


def init_weights(sess, w_path):
    """
    Loads official deep gaze 2 weights in network
    """
    debug = (0==1)
    # init weights
    weights_fn = 'meta/deep_gaze.npy'
    weights = np.load(weights_fn).item() # fuck pickles
    gt_var_list = [l.split("\n")[0] for l in open('meta/deep_gaze_var_names.txt').readlines()]
    me_var_list = [l.split("\n")[0] for l in open('meta/me_var_names.txt').readlines()]
    net_var_list =  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    for var in net_var_list:
        do_restore = (0==1)

        if var.op.name not in me_var_list:
            continue
        
        # me to deep_gaze (dg)
        if var.op.name.split("/")[-1] == 'kernel':
            do_restore = (1==1)
            dg_var_name_l = var.op.name.split("/")[:-1]
            dg_var_name = ''
            for elt in dg_var_name_l:
                dg_var_name += elt + '/'
            dg_var_name += 'weights'
            if debug:
                print('%s -> %s'%(var.op.name, dg_var_name))

        if var.op.name.split("/")[-1] == 'bias':
            do_restore = (1==1)
            dg_var_name_l = var.op.name.split("/")[:-1]
            dg_var_name = ''
            for elt in dg_var_name_l:
                dg_var_name += elt + '/'
            dg_var_name += 'biases'
            if debug:
                print('%s -> %s'%(var.op.name, dg_var_name))


        if ('alpha' in var.op.name or 'sigma' in var.op.name):
            #print(var)
            do_restore = (1==1)
            dg_var_name = var.op.name
            weights[dg_var_name] = np.expand_dims(weights[dg_var_name],0)
            if debug:
                print('%s -> %s'%(var.op.name, dg_var_name))
        
        if do_restore:
            sess.run(var.assign(weights[dg_var_name]))
        else:
            print('Not restored: %s'%var.op.name)

def get_activation_fn(act_type='relu', **kwargs):
    """
    Copied from lf-net: https://github.com/vcg-uvic/lf-net-release 
    """
    act_type = act_type.lower()
    act_fn = None
    if act_type == 'relu':
        act_fn = tf.nn.relu
    elif act_type == 'leaky_relu':
        alpha = kwargs.pop('alpha', 0.2)
        act_fn = lambda x, name=None : tf.nn.leaky_relu(x, alpha, name=name)
    elif act_type == 'sigmoid':
        act_fn = tf.nn.sigmoid
    elif act_type == 'tanh':
        act_fn = tf.nn.tanh
    elif act_type == 'crelu':
        act_fn = tf.nn.crelu
    elif act_type == 'elu':
        act_fn = tf.nn.elu
    else:
        print('Error: unknow activation function: %s'%act_fn)
        exit(1)

    print('Act-Fn: ', act_fn)
    return act_fn


