
import os
import argparse
import h5py

import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors
from PIL import Image

MACHINE = 0

class Metadata():
    """Metadata holder for the datasets."""

    def __init__(self, split_dir):
        """Load metadata from file.

        Args:
            dbImage: relative database filenames.
            utmDb: camera poses for database images (not necessarily utm).
            qImage: relative query filenames.
            utmQ: camera poses for query images.
            dist_pos: maximum distance between matching db and q images.
            dist_non_neg: minimum distance between non-matching db and q imgs.
        """
        self.dbImage = np.loadtxt('%s/dbImage.txt'%split_dir, dtype=str)
        self.utmDb = np.loadtxt('%s/utmDb.txt'%split_dir)
        self.qImage = np.loadtxt('%s/qImage.txt'%split_dir, dtype=str)
        self.utmQ = np.loadtxt('%s/utmQ.txt'%split_dir)
        meta = np.loadtxt('%s/meta.txt'%split_dir)
        self.dist_pos = meta[0]
        self.dist_non_neg = meta[1]

        self.numDb = self.dbImage.shape[0]
        self.numQ = self.qImage.shape[0]


def load_img(img_fn, new_size, mean, std):
    """Reads image from file and process it.

    Args:
        img_fn: full path filename.
        new_size: new image size.
        mean: BGR image to substract from the image.
        std: BGR standard-deviation to normalise the image.
    """
    if MACHINE==2:
       img = Image.open(img_fn) 
       if new_size is not None:
           img = np.array(img.resize(new_size))[:,:,::-1]
       else:
           img = np.array(img)[:,:,::-1]
       #print(sal.shape)
    else:
       #print(img_fn)
       img = cv2.imread(img_fn)
       if new_size is not None:
           #print(new_size)
           img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
           #img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    img = (img.astype(np.float32) - mean)/std
    return img

   

class ImageDataset(object):
    """Wrapper to sample dataset images and the ground-truth matching between
    q and db images. """

    def __init__(self, args, split_dir, is_training, onlyDB=False):
        """ Creates a dataset instance.

        Args:
            args: misc variables
            split_dir: dir where the dataset filenames and poses are stored.
        """
        self.ptr = 0
        
        self.batch_size = args.batch_size
        self.resize = args.resize 
        self.new_size = (args.w, args.h)
        self.is_training = is_training
        
        mean_std = np.loadtxt(args.mean_fn)
        self.mean = mean_std[0,:]*255.
        self.std = mean_std[1,:]*255.

        self.metadata = Metadata(split_dir)
        self.img_fn_l = ['%s/%s'%(args.img_dir, dbIm) for dbIm in self.metadata.dbImage]
        if not onlyDB:
            self.img_fn_l += ['%s/%s'%(args.img_dir, qIm) for qIm in self.metadata.qImage]
        self.dataset_size = len(self.img_fn_l)
        self.idx = np.arange(self.dataset_size)

        self.positives = None
        self.distances = None
        

    def next_batch(self):
        """Get a batch of images.
        
        Returns:
            new_epoch: 1 if a new epoch stars.
            np.array(img_l): [batch_size, h, w, c] np array. Image batch.
            np.array(idx_l): [batch_size]. Image batch index.
        """
        new_epoch = 0
        if self.dataset_size <=self.ptr + self.batch_size:
            if self.is_training:
                np.random.shuffle(self.idx)
            self.ptr = 0
            new_epoch = 1
            #print('new epoch: ptr= %d' %(self.train_ptr))
        
        img_l, idx_l = [], []
        for i in range(self.batch_size):
            #print('img_fn: %s'%self.img_fn_l[self.ptr])
            img = load_img(self.img_fn_l[self.ptr], self.new_size, self.mean, self.std)
            img_l.append(img)
            idx_l.append(self.ptr)
            self.ptr += 1
        return new_epoch, np.array(img_l), np.array(idx_l)

    
    def size(self):
        return self.dataset_size


    def getPositives(self):
        """Computes matching database images for each query.

        Returns:
            positives: array of array where positives[i] holds the list of db 
                img idx that match the i-th q img.
        """
        # positives for evaluation are those within trivial threshold range
        # fit NN to find them, search by radius
        if  self.positives is None:
            knn = NearestNeighbors(n_jobs=1)
            knn.fit(self.metadata.utmDb)
            self.distances, self.positives = knn.radius_neighbors(self.metadata.utmQ,
                    radius=self.metadata.dist_pos)
        return self.positives


class QueryDataset(object):
    """Query sampler that implements hard-negative mining."""

    def __init__(self, args, split_dir, is_training, cache_size, debug=False):
        """Creates a query sample instance.
        Args:
            args: misc variables
            split_dir: dir where the dataset filenames and poses are stored.
            is_training: set to true if training
            cache_size: [img_num, des_dim]
            N_nr: random negatives from which you compute the hardest negatives
            N_nh: number of hard negatives in the loss
            margin: violation margin in the triplet loss (m variable in the paper)
        """
        self.ptr = 0
        self.batch_size = args.batch_size
        self.resize = args.resize 
        self.new_size = (args.w, args.h)
        self.is_training = is_training
        self.img_dir = args.img_dir
        self.margin = args.margin
        self.N_nr = args.N_nr # negative random, number of negatives to randomly sample, N_nr
        self.N_nh = args.N_nh # number of negatives used for training N_nh

        mean_std = np.loadtxt(args.mean_fn)
        self.mean = mean_std[0,:]*255.
        self.std = mean_std[1,:]*255.

        self.metadata = Metadata(split_dir)
        self.dist_pos = self.metadata.dist_pos
        self.dataset_size = self.metadata.numQ
        self.idx = np.arange(self.dataset_size)

        # Compute potential positives
        # there are no trivial positives in the lake dataset between the query
        # and the database are disjoint
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.metadata.utmDb)

        # list of array of db idx matching a query i.e. with ||db - q||_2 < self.dist_pos
        # positives[i] = list of db img idx matching the i-th query
        if not debug:
            self.positives = list(knn.radius_neighbors(self.metadata.utmQ,
                    radius=self.dist_pos, return_distance=False))
            # radius returns unsorted, sort once now so we dont have to later
            # and yes this is needed because h5feat[self.nontrivial_positives[index].tolist()]
            # below requires the index to be in sorting order
            for i,posi in enumerate(self.positives):
                self.positives[i] = np.sort(posi)

        # visual check that positives are positives
        else:  
            # DEBUG this motherfucker
            debug_dist_pos, self.positives = knn.radius_neighbors(self.metadata.utmQ,
                    radius=self.dist_pos, return_distance=True)

            for i, pos_idx_v in enumerate(self.positives):
                print('\nquery: %d'%i)
                print('pos_idx_v: ', pos_idx_v)
                q_img_fn = '%s/%s'%(self.img_dir, self.metadata.qImage[i])
                q_img = cv2.imread(q_img_fn)
                q_xy = self.metadata.utmQ[i,:]

                for j, pos_idx in enumerate(pos_idx_v):
                    db_img_fn = '%s/%s'%(self.img_dir, self.metadata.dbImage[pos_idx])
                    db_img = cv2.imread(db_img_fn)
                    db_xy = self.metadata.utmDb[pos_idx,:]

                    d = np.sqrt(np.sum( (db_xy - q_xy)**2))

                    cv2.imshow('q | db', np.hstack((q_img, db_img)))
                    #print(pos_idx)
                    #print(debug_dist_pos[i][j])
                    #print(d, q_xy[0], q_xy[1])
                    print('KNN: pos_idx: %d\tdist: %.5f\tME: dist: %.3f\tq_xy/db_xy: %.3f / %.3f\t %.3f / %.3f'
                            %(pos_idx, debug_dist_pos[i][j], d, q_xy[0], q_xy[1], db_xy[0], db_xy[1]))
                    stop = cv2.waitKey(0) & 0xFF
                    if stop == ord("q"):
                        exit(0)

                #if i==10:
                #    break
            exit(0)

        # potential negatives are those outside of dist_pos range
        # since my lake is a continuous environment, I can not define
        # negatives as not positives since the images will overlap
        non_neg_ll = list(knn.radius_neighbors(self.metadata.utmQ,
                radius=self.metadata.dist_non_neg, return_distance=False))
        self.negatives = [] # negatives[i] = array of db img idx not matching the i-th query
        for non_neg_l in non_neg_ll:
            self.negatives.append( np.setdiff1d(np.arange(self.metadata.numDb),
                non_neg_l, assume_unique=True))
        

        # filepath of HDF5 containing feature vectors for images
        self.cache_fn = 'res/%d/cache/train_feat_cache.npy'%args.trial
        self.cache = np.zeros(cache_size, np.float32)
        self.cache_birthday = 0
        self.is_cache_init = 0

        # negCache[i] = list of hardest negative for i-th query at previous epoch
        self.negCache = [np.empty((0,)) for _ in range(self.metadata.numQ)]
        
        # current loss for each query (used for importance sampling) when the
        # sample query does not violate the margin
        # TODO: I may need to counter balance the sampling bias with the beta
        # variable.
        self.q_losses = np.zeros(self.dataset_size)
        self.past_db_idx_pos = np.zeros(self.dataset_size, np.int32) # idx of the nearest db img


    def next_item(self, index):
        #print('index: %d'%index)

        if self.is_cache_init == 0:
            self.cache = np.load(self.cache_fn)
            self.is_cache_init = 1
        else: # check modification date
            current_birthday = os.stat(self.cache_fn).st_ctime
            if current_birthday > self.cache_birthday:
                self.cache_birthday = current_birthday
                self.cache = np.load(self.cache_fn)
 
        q_feat = self.cache[ self.metadata.numDb + index,: ] # query featurei

        # sample >0 examle i.e. get nearest matching db img descriptor
        db_matching_feat = self.cache[self.positives[index],:] # features of matching db imgs
        knn = NearestNeighbors(n_jobs=-1) # TODO replace with faiss?
        knn.fit(db_matching_feat)

        # find nearest img in descriptor space among the matching db img
        nn_dist_pos, nn_idx = knn.kneighbors(q_feat.reshape(1,-1), 1) # find nearest db img and its dist
        nn_dist_pos = nn_dist_pos.item() # feature distance between q and its nearest db matching img
        db_idx_pos = self.positives[index][nn_idx[0]].item() # db idx of the nearest db matching img 
        self.past_db_idx_pos[index] = db_idx_pos # update the nearest matching db img idx
        #print('nn_idx: ', nn_idx)
        #print('db_idx_pos: %d'%db_idx_pos)


        # sample <0 examples
        # randomly sample N_nr negatives
        db_idx_rand_neg = np.random.choice(self.negatives[index], self.N_nr) # idx of db random neg
        # mix them with the N_nh hard-neg examples of the previous epoch
        db_idx_rand_neg = np.unique(np.concatenate([self.negCache[index], db_idx_rand_neg]))
        db_num_neg = db_idx_rand_neg.shape[0] # number of negatives to sample from
        
        # look for the N_nh hardest negatives
        db_feat_neg = self.cache[db_idx_rand_neg.astype(np.int),:]
        knn.fit(db_feat_neg)
        # look for 10*N_nh negatives instead of N_nh so that if there are
        # some that do not violate the constraints, I can discard them and
        # still have N_nh negative examples
        nn_dist_neg, nn_idx_neg = knn.kneighbors(q_feat.reshape(1,-1), min(10*self.N_nh, db_num_neg))
        nn_dist_neg = nn_dist_neg.reshape(-1)
        nn_idx_neg = nn_idx_neg.reshape(-1)
        

        # handle useless example that do not violate the margin
        violatingNeg = nn_dist_neg < nn_dist_pos + self.margin
        valid_idx_neg = np.where(nn_dist_neg < nn_dist_pos + self.margin)[0]
        
            
        # if you don't care about optimising 0 loss
        if (1==1):
            #print("I am big motherfucking gpu who likes to work for nothing (dab)")
            nn_idx_neg = nn_idx_neg[:self.N_nh] # keep only the violating ones
            db_idx_neg = db_idx_rand_neg[nn_idx_neg].astype(np.int32) # db idx of the N_nh hardest negative examples
            self.negCache[index] = db_idx_neg # update the N_nh hardest examples
        # load imgs
        q_img = load_img('%s/%s'%(self.img_dir, self.metadata.qImage[index]),
                self.new_size, self.mean, self.std)
        p_img = load_img('%s/%s'%(self.img_dir, self.metadata.dbImage[db_idx_pos]), 
                self.new_size, self.mean, self.std)
        
        n_img_l = [] # list of <0 img
        for idx_neg in db_idx_neg:
            n_img = load_img('%s/%s'%(self.img_dir, self.metadata.dbImage[idx_neg]), 
                self.new_size, self.mean, self.std)
            n_img_l.append(n_img)
        

        return q_img, p_img, n_img_l, [index, db_idx_pos] + db_idx_neg.tolist()

    
    def next_batch(self):
        """ 
        Get triplets
        """
        #print(self.pcl[self.train_idx[self.train_ptr]])
        new_epoch = 0
        if self.dataset_size <=self.ptr + self.batch_size:
            if self.is_training:
                np.random.shuffle(self.idx)
            self.ptr = 0
            new_epoch = 1
            #print('new epoch: ptr= %d' %(self.ptr))
        
        q_img_l, p_img_l, n_img_ll, idx_ll = [], [], [], []
        for i in range(self.batch_size):
            self.ptr += 1
            #print('self.ptr: %d'%self.ptr)
            q_img, p_img, n_img_l, idx_l = self.next_item(self.ptr)
            q_img_l.append(q_img)
            p_img_l.append(p_img)
            n_img_ll += n_img_l
            idx_ll.append(idx_l)
            
        q_img_v = np.stack(q_img_l)
        p_img_v = np.array(p_img_l)
        n_img_v = np.array(n_img_ll)
        idx_v = np.array(idx_ll)

        return new_epoch, q_img_v, p_img_v, n_img_v, idx_v


    def size(self):
        return self.dataset_size



if __name__=='__main__':
    print("Test data")

    parser = argparse.ArgumentParser()
    parser.add_argument('--trial', type=int, required=True)
    parser.add_argument('--data_id', type=int, required=True)
    parser.add_argument('--split_dir', type=str, required=True)
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=24)

    parser.add_argument('--resize', type=int, default=0)
    parser.add_argument('--h', type=int, default=0)
    parser.add_argument('--w', type=int, default=0)
    
    parser.add_argument('--N_nr', type=int, default=100)
    parser.add_argument('--N_nh', type=int, default=10)
    parser.add_argument('--margin', type=float, default=0.1)
    
    parser.add_argument('--mean_fn', type=str, default='')

    args = parser.parse_args()

    
    mean_std = np.loadtxt(args.mean_fn)
    mean = mean_std[0,:]*255.
    std = mean_std[1,:]*255.

    # Load data
    print('** Loading datasets **')
    
    train_dir = '%s/%d/train/'%(args.split_dir, args.data_id)
    is_training = True
    img_dataset = ImageDataset(args, train_dir, is_training, onlyDB=False)
    if (0==1):
        _, img_batch = img_dataset.next_batch()
        for i in range(args.batch_size):
            img = img_batch[i,:,:,:]
            img = (img*std + mean).astype(np.uint8)
            cv2.imshow('img', img)
            cv2.waitKey(0)

    if (0==1):
        is_training = False
        next_epoch = 0
        query_dataset = QueryDataset(args, train_dir, is_training, debug=False)
        while next_epoch !=1:
            next_epoch, q_img_v, p_img_v, n_img_v, idx_v = query_dataset.next_batch()

            for i in range(args.batch_size):
                q_img = q_img_v[i,:,:,:]
                p_img = p_img_v[i,:,:,:]
                q_img = (q_img*std + mean).astype(np.uint8)
                p_img = (p_img*std + mean).astype(np.uint8)

                for j in range(args.N_nh):
                    #print(i+j)
                    n_img = n_img_v[i+j,:,:,:]
                    n_img = (n_img*std + mean).astype(np.uint8)
                    cv2.imshow('q | p | n', np.hstack((q_img, p_img, n_img)))
                    stop_show = cv2.waitKey(0)
                    if stop_show == ord("q"):
                        exit(0)

    
    if (1==1):
        # validation data
        val_dir = '%s/%d/val/'%(args.split_dir, args.data_id)
        img_dataset = ImageDataset(args, val_dir, is_training, onlyDB=False)
        
        gt = img_dataset.getPositives()
    
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(img_dataset.metadata.utmDb)
        distances, positives = knn.radius_neighbors(img_dataset.metadata.utmQ,
                radius=img_dataset.metadata.dist_pos)

        print(distances.shape)
        print(positives.shape)

        print(distances)
        print(positives)

        for i, (d_v, p_v) in enumerate(zip(distances, positives)):
            print('\nIter: %d'%i)
            print(d_v)
            print(p_v)

            dbImage_fn = '%s/%s'%(args.img_dir, img_dataset.metadata.dbImage[i])
            print('dbImage_fn: %s'%dbImage_fn)
            dbImage = cv2.imread(dbImage_fn)

            for j, (d,p) in enumerate(zip(d_v, p_v)):
                qImage_fn = '%s/%s'%(args.img_dir, img_dataset.metadata.qImage[p])
                print(qImage_fn)
                qImage = cv2.imread(qImage_fn)

                print('d: %.3f'%d)
                cv2.imshow('dbImage | qImage', np.hstack((dbImage, qImage)))
                cv2.waitKey(0)


