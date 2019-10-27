"""data feeder"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
from six.moves import xrange  # pylint: disable=redefined-builtin
import numpy as np
from math import sqrt, exp

class Dataloader(object):
    def __init__(self, img_dir, edge_dir, do_train, batch_size,
            patch_size, mean_fn):
        """
        Args:
            img_dir: img directory
            edge_dir: edge directory
            do_train: Set to 1 if you train
            batch_size: batch size
            patch_size: size of the patchs fed to the network 
            mean_fn: path to dataset mean file
        """
        self.img_dir = img_dir
        self.edge_dir = edge_dir
        self.batch_size = batch_size
        self.train = (do_train==1)
        self.patch_size = patch_size
        self.pad = 0 
        self.mean = np.loadtxt(mean_fn)
        
        self.train_id_v = np.loadtxt('meta/train.txt', dtype=str)
        self.val_id_v = np.loadtxt('meta/val.txt', dtype=str)

        self.dataset_size = self.train_id_v.shape[0]
        if do_train:
            self.idx = self.train_id_v
        else:
            self.idx = self.val_id_v
        self.ptr = 0
        self.idx_batch = []
        if self.train:
            np.random.shuffle(self.idx)


    def next_batch(self):
        """ 
        Get img pair and label 
        """
        pad = self.pad
        new_epoch = 0
        patch_w, patch_h = self.patch_size
        
        patch_batch, gt_patch_batch = [],[]
        for i in range(self.batch_size):

            if self.dataset_size == self.ptr:
                if self.train:
                    np.random.shuffle(self.idx)
                self.ptr = 0
                new_epoch = 1
                #print('new epoch: ptr= %d' %(self.train_ptr))
            
            self.idx_batch.append(self.idx[self.ptr]) # instance id (for visu valid purposes)
            img_id = self.idx[self.ptr]

            #print(os.path.join(self.img_dir, '%s'%img_id))
            img = cv2.imread(os.path.join(self.img_dir, '%s'%img_id))
            gt = cv2.imread(os.path.join(self.edge_dir, '%s'%img_id),
                    cv2.IMREAD_UNCHANGED)

            h,w = img.shape[:2]

            # random crop patch
            bi = np.random.randint(h-patch_h-self.pad-1)
            ei = bi + patch_h
            bj = np.random.randint(w-patch_w-self.pad-1)
            ej = bj + patch_w

            patch   = img[bi:ei, bj:ej]
            gt_patch = gt[bi:ei, bj:ej]

            # post proc
            patch = patch.astype(np.float32) - self.mean
            gt_patch = (gt_patch.astype(np.float32)/255.)
            #gt_patch = (gt_patch.astype(np.float32)/255).astype(np.uint8)
            
            
            #print(np.unique(gt_patch))
            #cv2.imshow('gt', gt_patch.astype(np.float32))
            #cv2.waitKey(0)
            #exit(0)
            gt_patch = np.expand_dims(gt_patch, 2)

            patch_batch.append(patch)
            gt_patch_batch.append(gt_patch)
            

            self.ptr += 1
            #print('self.train_ptr: %d' %(self.train_ptr))
        return new_epoch, np.stack(patch_batch), np.stack(gt_patch_batch)

