"""data feeder"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from math import sqrt, exp
import numpy as np
from scipy.misc import logsumexp

from tools.cst import *
if MACHINE==2:
    from PIL import Image
else:
    import cv2


class Dataloader(object):
    def __init__(self, csv_file, data_dir, batch_size, out_size, resize_img, mean_file, is_training):
        """
        Args:
            data_dir: path to the hpatch dataset
            csv_file: file with list of hpatch img name
            mean_file: path to image mean file
            out_size: (new_width, new_height)
            max_img_num: number of image per image directory in [1,6]
        """
        self.data = [l.split("\n")[0] for l in open(csv_file, 'r').readlines()]
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.out_size = out_size
        self.is_training = is_training
        self.mean = np.loadtxt(mean_file) # bgr mean

        self.dataset_size = len(self.data)
        self.idx = np.arange(self.dataset_size)
        if self.is_training:
            np.random.shuffle(self.idx)
        self.ptr = 0
        self.idx_batch = []
        self.img_num = 1
        self.resize_img = resize_img
    

    def next_batch(self):
        """ 
        Get img pair and label 
        """
        #print(self.pcl[self.train_idx[self.train_ptr]])
        new_epoch = 0
        if self.dataset_size <=self.ptr + self.batch_size:
            if self.is_training:
                np.random.shuffle(self.idx)
            self.ptr = 0
            new_epoch = 1
            #print('new epoch: ptr= %d' %(self.train_ptr))
        
        img_batch_l, sal_batch_l = [],[]
        for i in range(self.batch_size):
            img_root_fn, sal_root_fn = self.data[self.ptr].split(" ")
            img_fn = os.path.join(self.data_dir, img_root_fn)
            sal_fn = os.path.join(self.data_dir, sal_root_fn)
            #print('img_fn: %s' %(img_fn))
            if MACHINE==2:
                img = Image.open(img_fn) 
                sal = Image.open(sal_fn)
                if self.resize_img:
                    img = np.array(img.resize(self.out_size))[:,:,::-1]
                    sal = np.array(sal.resize(self.out_size)).astype(np.float32)
                else:
                    img = np.array(img)[:,:,::-1]#.astype(np.float32)
                    sal = np.array(sal)#.astype(np.float32)
                #print(sal.shape)
            else:
                img = cv2.imread(img_fn)
                sal = cv2.imread(sal_fn, cv2.IMREAD_UNCHANGED)
                if self.resize_img:
                    img = cv2.resize(img, self.out_size, interpolation=cv2.INTER_AREA)
                    sal = cv2.resize(sal, self.out_size, interpolation=cv2.INTER_AREA)


            # pre-proc img
            img = img.astype(np.float32) - self.mean
            
            # pre proc saliency
            # scale to [0.5,1]
            sal = sal.astype(np.float32)
            old_min, old_max = np.min(sal), np.max(sal)
            new_min, new_max = 0.5, 1.0
            sal = new_min + (sal - old_min) * (new_max - new_min) / (old_max - old_min)
            # convert into log prob
            sal -= logsumexp(sal) # log(softmax(sal))

            sal = np.expand_dims(sal, 2)
            img_batch_l.append(img)
            sal_batch_l.append(sal)

            self.ptr += 1
        #print('self.train_ptr: %d' %(self.train_ptr))
        #print(np.array(img_batch_l).shape)
        #print(np.array(sal_batch_l).shape)
        return new_epoch, np.array(img_batch_l), np.array(sal_batch_l)

