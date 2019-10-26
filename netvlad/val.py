"""Evaluates the trained netvlad with recall@N and mAP."""
import os, shutil, json, argparse, random, time
from datetime import datetime
import h5py
import faiss

import numpy as np
import tensorflow as tf

import tools.data_loader # dataloader
import tools.opt as opt # loss, optimiser
from tools.model_netvlad import vgg16NetvladPca as net

netvlad_ckpt_dir = 'meta/weights/netvlad_tf_open/vd16_pitts30k_conv5_3_vlad_preL2_intra_white'

def val(args):
    bz = args.batch_size
    new_size = (args.w, args.h)
    
    val_log_dir = 'res/%d/log/val'%args.trial

    print('** Load data **')
    is_training = False
    val_dir = '%s/%d/val/'%(args.split_dir, args.data_id)
    img_dataset = tools.data_loader.ImageDataset(args, val_dir, is_training, onlyDB=False)
    img_num = img_dataset.size()
    print('# val images: %d'%img_num)

    with tf.Graph().as_default():
        # define network operations graph
        img_op = tf.placeholder(dtype=tf.float32, shape=[bz, new_size[1], new_size[0], 3])
        tf.summary.image('img', img_op)
        des_op = net(img_op)
        print('img_op.shape: ', img_op.get_shape())
        print('des_op.shape: ', des_op.get_shape())

        # init netvlad with paper weights
        var_to_init = []
        var_to_init_name = list(np.loadtxt('meta/var_to_init_netvlad.txt', dtype=str))
        all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        for var in all_vars:
            if var.op.name in var_to_init_name:
                var_to_init.append(var)
        saver_init = tf.train.Saver(var_to_init)
        
        # Set saver to restore finetuned network 
        variable_averages = tf.train.ExponentialMovingAverage(args.moving_average_decay)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Set summary op, restore vars
        summary_op = tf.summary.merge_all()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if args.no_finetuning == 1: # load model from pittsburg trainng
                global_step = 0
                print("Evaluate netvlad from the paper")
                ckpt = tf.train.get_checkpoint_state(netvlad_ckpt_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    print("checkpoint path: ", ckpt.model_checkpoint_path)
                    saver_init.restore(sess, ckpt.model_checkpoint_path)
                else:
                    print('No checkpoint file found at: %s'%netvlad_ckpt_dir)
                    return
                print('Load model Done')
            else:
                print("Evaluate my super finetuned version")
                train_log_dir = 'res/%d/log/train'%args.trial
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
                summary_writer = tf.summary.FileWriter(val_log_dir, graph_def=sess.graph_def)
            

                # store all img feature in an array
                des_dim = des_op.get_shape().as_list()[-1]
                des_v = np.empty((img_num, des_dim))
                next_epoch = 0
                while next_epoch != 1:
                    next_epoch, img, idx = img_dataset.next_batch()
                    start_time = time.time()
                    des = sess.run(des_op, feed_dict={img_op: img})
                    duration = time.time() - start_time
                    
                    des_v[idx, :] = des 
                    
                
                ## Copy-pasted from https://github.com/Nanne/pytorch-NetVlad
                # extracted for both db and query, now split in own sets
                q_des_v = des_v[img_dataset.metadata.numDb:].astype(np.float32)
                db_des_v = des_v[:img_dataset.metadata.numDb].astype(np.float32)

                print('Building faiss index')
                faiss_index = faiss.IndexFlatL2(des_dim)
                faiss_index.add(db_des_v)
                n_values = [1,5,10,20]
                _, predictions = faiss_index.search(q_des_v, max(n_values)) 

                # for each query get those within threshold distance
                gt = img_dataset.getPositives() 

                print('Calculating recall @ N')
                correct_at_n = np.zeros(len(n_values))
                #TODO can we do this on the matrix in one go?
                for qIx, pred in enumerate(predictions):
                    for i,n in enumerate(n_values):
                        # if in top N then also in top NN, where NN > N
                        if np.any(np.in1d(pred[:n], gt[qIx])):
                            correct_at_n[i:] += 1
                            break
                recall_at_n = correct_at_n / img_dataset.metadata.numQ

                recalls = {} #make dict for output
                for i,n in enumerate(n_values):
                    recalls[n] = recall_at_n[i]
                    print("Recall@%d: %.4f"%(n, recall_at_n[i]))

                summary = tf.Summary()
                summary.ParseFromString(sess.run(summary_op, feed_dict={img_op: img}))
                for i,n in enumerate(n_values):
                    summary.value.add(tag='Recall@%d'%n, simple_value=recall_at_n[i])
                summary_writer.add_summary(summary, global_step)

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
   
    # optim 
    parser.add_argument('--model', type=str, default='', help='{alexnet, vgg}')
    parser.add_argument('--num_clusters', type=int, default=64, help='Number of clusters')

    parser.add_argument('--moving_average_decay', type=float, default=0.9999, help='')
    # log
    parser.add_argument('--no_finetuning', type=int, help='Set to 1 if you start train.')
    
    args = parser.parse_args()
    val(args)

