import os
import numpy as np

def netvlad_gen_dataset(data_id, splits_d, whichSet):
    # TODO: handle the cameras for the negative sampling
    cam_id = 0

    # TODO: refine these values
    # posDistThr: distance in meters which defines positives i.e. matching
    # images
    dist_pos = 5 # meters
    # distance above which images are not matching
    dist_non_neg = 10 # meters

    out_dir = 'meta/data_splits/cmu/%d/%s'%(data_id, whichSet)
    if os.path.exists(out_dir):
        answer = input("This dataset already exists. Do you want to delete it ?: %s\n"%(out_dir))
        if answer=='n':
            exit(0)
    else:
        os.makedirs(out_dir)

    fn_d, utm_d = {}, {}
    for split in ['db', 'q']:
        fn_d[split], utm_d[split] = [], []

    for slice_id in splits_d[whichSet]['slices']:
        for split in ['db', 'q']:
            for survey_id in splits_d[whichSet][slice_id][split]:
                if survey_id == -1:
                    meta_fn = 'meta/data/cmu/surveys/%d/%d_c%d_db/pose.txt'%(
                            slice_id, slice_id, cam_id)
                else:
                    meta_fn = 'meta/data/cmu/surveys/%d/%d_c%d_%d/pose.txt'%( 
                            slice_id, slice_id, cam_id, survey_id)

                if os.stat(meta_fn).st_size==0:
                    continue # ground truth pose are not available

                meta = np.loadtxt(meta_fn, dtype=str)
                img_v = meta[:,0]
                pose_v = meta[:,5:7].astype(np.float32) # take only x,y

                # sample 1/2 image
                img_num = img_v.shape[0]
                img_v = img_v[np.arange(0,img_num,2)]
                pose_v = pose_v[np.arange(0,img_num,2),:]

                print(img_v.shape)
                fn_d[split].append(img_v)
                utm_d[split].append(pose_v)
    
    np.savetxt('%s/dbImage.txt'%out_dir, np.hstack(fn_d['db']), fmt='%s')
    np.savetxt('%s/qImage.txt'%out_dir, np.hstack(fn_d['q']), fmt='%s')
    np.savetxt('%s/utmDb.txt'%out_dir, np.vstack(utm_d['db']))
    np.savetxt('%s/utmQ.txt'%out_dir, np.vstack(utm_d['q']))
    np.savetxt('%s/meta.txt'%out_dir, np.array([dist_pos, dist_non_neg]))


if __name__=='__main__':
    data_id = 0

    splits_d = {}
    splits_d['train'] = {}
    splits_d['train']['slices'] = [22,23] # specify which slices to use
    splits_d['train'][22] = {} # specify database/query sets
    splits_d['train'][22]['db'] = list(range(-1,5))
    splits_d['train'][22]['q'] = list(range(5,10))

    splits_d['train'][23] = {}
    splits_d['train'][23]['db'] = list(range(-1,5))
    splits_d['train'][23]['q'] = list(range(5,10))

    splits_d['val'] = {}
    splits_d['val']['slices'] = [24]
    splits_d['val'][24] = {}
    splits_d['val'][24]['db'] = list(range(-1,5))
    splits_d['val'][24]['q'] = list(range(5,10))
    
    splits_d['test'] = {}
    splits_d['test']['slices'] = [25]
    splits_d['test'][25] = {}
    splits_d['test'][25]['db'] = list(range(-1, 5))
    splits_d['test'][25]['q'] = list(range(5,10))

    netvlad_gen_dataset(data_id, splits_d, 'train')
    netvlad_gen_dataset(data_id, splits_d, 'val')
    netvlad_gen_dataset(data_id, splits_d, 'test')
   
