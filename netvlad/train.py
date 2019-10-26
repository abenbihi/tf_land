"""Training script for netvlad."""
import argparse
from datetime import datetime
import os 
import random
import time
import h5py

import numpy as np
import tensorflow as tf

import tools.data_loader
import tools.opt
import tools.model_netvlad

netvlad_ckpt_dir = 'meta/weights/netvlad_tf_open/vd16_pitts30k_conv5_3_vlad_preL2_intra_white'


def update_cache(sess, q_img_op, q_des_cst_op, img_dataset, cache_fn):
    cache_start_time = time.time()
    img_num = img_dataset.size()
    des_dim = q_des_cst_op.get_shape().as_list()[-1]
    cache = np.zeros((img_num, des_dim), np.float32)
    
    img_count = 0 
    next_epoch = 0
    while next_epoch != 1:
        img_count += img_dataset.batch_size
        if img_count % 99 == 0:
            duration = time.time() - cache_start_time
            minutes = duration / 60
            seconds = duration % 60
            print('%d/%d %d:%02d'%(img_count, img_num, duration/60, duration%60))

        next_epoch, img, idx = img_dataset.next_batch()
        des = sess.run(q_des_cst_op, feed_dict={q_img_op: img})
        cache[idx,:] = des
        del des
        del img

    np.save(cache_fn, cache)
    del cache
    
    duration = time.time() - cache_start_time
    minutes = duration / 60
    seconds = duration % 60
    print('Cache updated in: %d:%02d'%(duration/60, duration%60))


def train(args):
    np.random.seed(args.seed)

    bz = args.batch_size
    N_nh = args.N_nh
    new_size = (args.w, args.h)
    
    train_log_dir = 'res/%d/log/train'%args.trial
    if not os.path.exists(train_log_dir):
        os.makedirs(train_log_dir)
    cache_fn = 'res/%d/cache/train_feat_cache.hdf5'%args.trial

    print('** Load data **')
    # train img (queries (q) and database (db))
    is_training = True
    train_dir = '%s/%d/train/'%(args.split_dir, args.data_id)
    img_dataset = tools.data_loader.ImageDataset(args, train_dir, is_training, onlyDB=False)
    print('# train images: %d'%img_dataset.size())
    # train queries
    img_num = img_dataset.size()
    des_dim = 4096 # yeah, yeah bad hardcoding, I know
    query_dataset = tools.data_loader.QueryDataset(args, train_dir, is_training,
            (img_num, des_dim), debug=False)
    print('# of train queries: %d' %query_dataset.size())


    # define network opts
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False) 

        q_img_op = tf.placeholder(dtype=tf.float32, shape=[bz, new_size[1], new_size[0], 3]) # query
        p_img_op = tf.placeholder(dtype=tf.float32, shape=[bz, new_size[1], new_size[0], 3]) # >0 examples
        n_img_op = tf.placeholder(dtype=tf.float32, shape=[bz * N_nh, new_size[1], new_size[0], 3]) # <0 examples
        print('q_img_op.shape: ', q_img_op.get_shape())
        print('p_img_op.shape: ', p_img_op.get_shape())
        print('n_img_op.shape: ', n_img_op.get_shape())

        # I prefer 3 forwards to stack for memory
        #feats_op = tf.concat([q_feat_op, p_feat_op, n_feat_op], axis=0)
        q_des_op = tools.model_netvlad.vgg16NetvladPca(q_img_op)
        p_des_op = tools.model_netvlad.vgg16NetvladPca(p_img_op, reuse=True)
        n_des_op = tools.model_netvlad.vgg16NetvladPca(n_img_op, reuse=True)
        print('q_des_op.shape: ', q_des_op.get_shape())
        print('p_des_op.shape: ', p_des_op.get_shape())
        print('n_des_op.shape: ', n_des_op.get_shape())
        q_des_cst_op = tf.stop_gradient(q_des_op)

        loss_op, loss_p_op, loss_n_op = tools.opttriplet_loss(args, q_des_op, p_des_op, n_des_op)
        var_to_train_name = list(np.loadtxt('meta/var_to_train_netvlad.txt', dtype=str))
        train_op = tools.opttrain(args, loss_op, global_step, var_to_train_name)

        # vars to init with paper weights
        var_to_init = []
        var_to_init_name = list(np.loadtxt('meta/var_to_init_netvlad.txt', dtype=str))
        all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        for var in all_vars:
            if var.op.name in var_to_init_name:
                var_to_init.append(var)
        #var_to_train = tf.trainable_variables()
        #print('\nvar to train')
        #for var in var_to_train:
        #    print(var.op.name)
        saver_init = tf.train.Saver(var_to_init)
        
        # Set saver to save/restore network for eval
        variable_averages = tf.train.ExponentialMovingAverage(args.moving_average_decay)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
   
        summary_op = tf.summary.merge_all()
        

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if args.start_train == 1: # init net with ImageNet pretraining
                global_step = 0
                ckpt = tf.train.get_checkpoint_state(netvlad_ckpt_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    print("checkpoint path: ", ckpt.model_checkpoint_path)
                    saver_init.restore(sess, ckpt.model_checkpoint_path)
                else:
                    print('No checkpoint file found in: \n%s'%netvlad_ckpt_dir)
                    return
                print('Weights initialization Done')
            else: # restart from your previous finetuning
                ckpt = tf.train.get_checkpoint_state(train_log_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    print("checkpoint path: ", ckpt.model_checkpoint_path)
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
                    threads.extend(qr.create_threads(sess, coord=coord,
                        daemon=True, start=True))
                summary_writer = tf.summary.FileWriter(train_log_dir, graph_def=sess.graph_def)
            
                current_epoch = 0
                step = int(global_step)
                if global_step == 0:
                    print('** Create cache **')
                    update_cache(sess, q_img_op, q_des_cst_op, img_dataset, cache_fn)
                
                while current_epoch < args.n_epochs:
                    step+=1
                    next_epoch, q_img_v, p_img_v, n_img_v, idx_v = query_dataset.next_batch()
                    start_time = time.time()
                    
                    _, loss, loss_p, loss_n = sess.run([train_op, loss_op,
                        loss_p_op, loss_n_op],
                            feed_dict={q_img_op: q_img_v, p_img_op: p_img_v, n_img_op: n_img_v})
                    assert not np.isnan(loss), 'Model diverged with loss = NaN'
                    duration = time.time() - start_time
                    
                    if step % args.C == 0:
                        os.remove('%s.npy'%cache_fn)
                        print('** Update cache **')
                        update_cache(sess, q_img_op, q_des_cst_op, img_dataset, cache_fn)

                    if step % args.log_interval == 0:
                        num_examples_per_step = args.batch_size
                        examples_per_sec = num_examples_per_step / duration
                        sec_per_batch = float(duration)
                        format_str = ('%s: epoch %d/%d, step %d, loss=%.3f, (%.1f examples/sec; %.3f ' 'sec/batch)')
                        print (format_str % (datetime.now(), current_epoch, args.n_epochs, step, 
                            loss, examples_per_sec, sec_per_batch))
                                          
                    if (step % args.summary_interval==0):
                        summary_str = sess.run(summary_op,
                            feed_dict={q_img_op: q_img_v, p_img_op: p_img_v, n_img_op: n_img_v})
                        summary_writer.add_summary(summary_str, step)
                    if next_epoch==1:
                        if (current_epoch != 0 and current_epoch % args.save_interval==0):
                            print('Save model')
                            checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                            saver.save(sess, checkpoint_path, global_step=step)
                        current_epoch +=1
                        
                    ## debug
                    #if step %10==0:
                    #    break # TODO: delete
                   

                # Save model
                summary_str = sess.run(summary_op,
                            feed_dict={q_img_op: q_img_v, p_img_op: p_img_v, n_img_op: n_img_v})
                summary_writer.add_summary(summary_str, step)
                checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

            except Exception as e:  # pylint: disable=broad-except
                coord.request_stop(e)
            
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)
           

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='pytorch-NetVlad')
    
    parser.add_argument('--trial', type=int, required=True, help='Trial.')

    # dataset
    parser.add_argument('--data_id', type=int, default=1)
    parser.add_argument('--mean_fn', type=str, default='', help='Path to mean/std.')
    parser.add_argument('--split_dir', type=str, default='', help='dataset split directory')
    parser.add_argument('--img_dir', type=str, default='', help='img directory')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--resize', type=int, default=0, help='set to 1 to resize img')
    parser.add_argument('--h', type=int, default=480, help='new height')
    parser.add_argument('--w', type=int, default=704, help='new width')
    parser.add_argument('--N_nr', type=int, default=10, help='Number of random negatives to sample.')
    parser.add_argument('--N_nh', type=int, default=10, help='Number of hard negatives.')
   
    # optim 
    parser.add_argument('--num_clusters', type=int, default=64, help='Number of clusters')
    parser.add_argument('--margin', type=float, default=0.1, help='Margin for triplet loss. Default=0.1')
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--C', type=int, default=1000, help='Cache update frequency (in batch)')
    parser.add_argument('--optim', type=str, help='optimizer', choices=['SGD', 'ADAM'])

    parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate.')
    parser.add_argument('--moving_average_decay', type=float, default=0.9999, help='')
    
    # adam
    parser.add_argument('--adam_b1', type=float, default=0.9, help='adam beta1')
    parser.add_argument('--adam_b2', type=float, default=0.999, help='adam beta2')
    parser.add_argument('--adam_eps', type=float, default=1e-08, help='adam epsilon')
    # SGD
    parser.add_argument('--num_epochs_per_decay', type=float, default=5, help='Decay LR ever N steps.')
    parser.add_argument('--lr_decay_factor', type=float, default=0.5, help='Multiply LR by Gamma for decaying.')
    parser.add_argument('--weightDecay', type=float, default=0.001, help='Weight decay for SGD.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD.')
    
    # log
    parser.add_argument('--log_interval', type=int, default=10, help='Log to stdout every such iteration.')
    parser.add_argument('--summary_interval', type=int, default=10, help='Log tensorboard every such iteration.')
    parser.add_argument('--save_interval', type=int, default=10, help='Save model every such epoch.')
    
    parser.add_argument('--seed', type=int, default=123, help='Random seed to use.')
    parser.add_argument('--start_train', type=int, help='Set to 1 if you start train.')
    
    args = parser.parse_args()
    train(args)

