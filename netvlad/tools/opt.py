

import tensorflow as tf


#def repeat(tensor, times):
#    """
#    Shamelessly copy-pasted from 
#    """
#
#    tensor = tf.expand_dims(tensor, axis=-1)
#    tensor = tf.tile(tensor, times)
#    print('post_tile: ', tensor.get_shape())
#    tensor = tf.reshape(tensor, shape=[-1])
#    return tensor

def triplet_loss(args, q_des_op, p_des_op, n_des_op):
    """
        Triplet loss
    """
    margin = args.margin

    # distance with >0 example
    dp = tf.reduce_sum(tf.square(q_des_op - p_des_op), (1))
    #dp = tf.reduce_sum(tf.abs(q_des_op - p_des_op), (1) # better ?
    #tf.summary.scalar('dp', dp)
    
    # distance with <0 example
    # repeat q_des_op to align with n_des_op
    # there is probably a better way to do this 
    # https://stackoverflow.com/questions/51822211/tensorflow-how-to-tile-a-tensor-that-duplicate-in-certain-order
    q_des_op = tf.expand_dims(q_des_op, axis=-1)
    q_des_op = tf.tile(q_des_op, [1, 1, args.N_nh])
    dims = q_des_op.get_shape().as_list()
    q_des_op = tf.reshape(q_des_op, shape = [dims[0] * dims[2], -1])
    dn = tf.reduce_sum(tf.square(q_des_op - n_des_op), (1))
    #tf.summary.scalar('dn', dn)
    
    # repeat dp to align with dn
    dp = tf.expand_dims(dp, axis=-1)
    dp = tf.tile(dp, [1,args.N_nh])
    dp = tf.reshape(dp, shape = [-1])
    
    loss_b = tf.maximum(0.0, margin + dp - dn) # loss batch
    #tf.summary.scalar('loss_b', loss_b)

    loss = tf.reduce_mean(loss_b) 
    tf.summary.scalar('loss', loss)
    
    tf.add_to_collection('losses', loss)
    # The total loss is defined as the l2 loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss'), dp, dn


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


def train(args, total_loss, global_step, var_to_train_name):
    """Train  model.
    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.
    Args:
      total_loss: Total loss from loss().
      global_step: Integer Variable counting the number of training steps
        processed.
    Returns:
      train_op: op for training.
    """

    # Variables that affect learning rate.
    num_batches_per_epoch = 545 / args.batch_size
    decay_steps = int(num_batches_per_epoch * args.num_epochs_per_decay)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(args.lr,
                                    global_step,
                                    decay_steps,
                                    args.lr_decay_factor,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    #var_to_train = tf.trainable_variables()
    #print('\nvar to train')
    #for var in var_to_train:
    #    print(var.op.name)

    # Set variables to optimize
    var_to_train = []
    for var in tf.trainable_variables():
        if var.op.name in var_to_train_name:
            var_to_train.append(var)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        
        #opt = tf.train.AdamOptimizer(args.lr, args.adam_b1, args.adam_b2, args.adam_eps)
        
        ## TODO: uncomment if your net has BN
        #update_ops =  tf.get_collection(tf.GraphKeys.UPDATE_OPS) #line for BN
        #with tf.control_dependencies(update_ops):
        #    grads = opt.compute_gradients(total_loss, var_list=var_to_train)
        #    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    
        # no BN
        grads = opt.compute_gradients(total_loss, var_list=var_to_train)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(args.moving_average_decay, global_step)

    with tf.control_dependencies([apply_gradient_op]):
        variables_averages_op = variable_averages.apply(var_to_train)
        #variables_averages_op = variable_averages.apply(tf.trainable_variables())
        #train_op = tf.no_op(name='train')

    return variables_averages_op #train_op


