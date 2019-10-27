
import os
import cv2
import numpy as np

slice_id = 6
caffe_dir = "/home/gpu_user/assia/ws/tf/hed/examples/hed/res/cmu/slice%d/fuse/"%slice_id
tf_dir = "res/cmu/slice%d/fuse/"%slice_id

for root_fn in sorted(os.listdir(tf_dir)):
    caffe_fn = "%s/%s"%(caffe_dir, root_fn)
    tf_fn = "%s/%s"%(tf_dir, root_fn)

    print(caffe_fn)
    print(tf_fn)

    caffe_img = cv2.imread(caffe_fn, cv2.IMREAD_UNCHANGED)
    tf_img = cv2.imread(caffe_fn, cv2.IMREAD_UNCHANGED)

    error = np.sum(caffe_img - tf_img)
    if error != 0:
        raise ValueError("Your tf code is wrong")
