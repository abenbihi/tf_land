
import os
import numpy as np


import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from collections import OrderedDict


MACHINE=2
SHOW    = (0==1)
WRITE   = (0==1)

# Train setting
ADD_CENTERBIAS = (1==1)


# Image dimensions
DEEP_GAZE_W, DEEP_GAZE_H = 1024,1024

# kreiman first resized the cropped waldo to this size before giving it to the
# subjects
RAW_CROPPED_W, RAW_CROPPED_H = 1280,1024

# I resized full image waldo to this size before locating waldo and his
# friends
GT_W, GT_H = 1500, 900


# Max image size user-friendly on the 27' screen
SCREEN_SIZE = 25 # pouces
if SCREEN_SIZE==27:
    WEB_W, WEB_H = 2200,1250 # big screen
elif SCREEN_SIZE==25:
    WEB_W, WEB_H = 1550,840 # laptop screen
else:
    print('Error: specify SCREEN_SIZE macro in {25,27}')
    exit(1)

# web display macro
MARGIN_LEFT = 340 # TODO test # image is shifted to the right
MARGIN_TOP = 70 # TODO: test # img is shifted from the top 


if MACHINE==0:
    HOME_DIR = '/home/abenbihi/ws/' 
elif MACHINE==1: # local gpu
    HOME_DIR = '/home/gpu_user/assia/ws/' 
elif MACHINE==2: # supelec gpu
    HOME_DIR = '/opt/BenbihiAssia/'
else:
    print('Error: you need to specify the MACHINE macro. Abort.')
    exit(1)

#GPU = (1==1)
#if GPU:
#    HOME_DIR = '/home/gpu_user/assia/ws/'
#else:
#    HOME_DIR = '/home/abenbihi/ws/' 

# data tools dir
TOOLS_DIR = os.path.join(HOME_DIR, 'tools/waldo/')

# web workspace
WEB_DIR = os.path.join(HOME_DIR, 'tools/eyetracker/WebGazer/www/')
WEB_DATA_DIR = os.path.join(WEB_DIR, 'media/example/')

# kreiman data
DATA_DIR        = os.path.join(HOME_DIR, 'datasets/waldo/')
RAW_DIR         = os.path.join(DATA_DIR, 'kreiman/')
RAW_IMG_DIR     = os.path.join(RAW_DIR, 'stimuli/')
RAW_GT_DIR      = os.path.join(RAW_DIR, 'gt/')
RAW_SCAN_DIR    = os.path.join(RAW_DIR, 'psy/ProcessScanpath_waldo/')
RAW_FULL_IMG_DIR= os.path.join(RAW_DIR, 'fullImage/')


#subj_l_fn = os.path.join(DATA_DIR, 'raw/subj_l.txt')
#subj_l = [l.split("\n")[0] for l in open(subj_l_fn, 'r').readlines()]

#img_l_fn = os.path.join(DATA_DIR, 'raw/img_fn_l.txt')
#img_l = [l.split("\n")[0] for l in open(img_l_fn, 'r').readlines()]

IMG_NUM = 134

myjet = np.array([[0.        , 0.        , 0.5       ],
                  [0.        , 0.        , 0.99910873],
                  [0.        , 0.37843137, 1.        ],
                  [0.        , 0.83333333, 1.        ],
                  [0.30044276, 1.        , 0.66729918],
                  [0.66729918, 1.        , 0.30044276],
                  [1.        , 0.90123457, 0.        ],
                  [1.        , 0.48002905, 0.        ],
                  [0.99910873, 0.07334786, 0.        ],
                  [0.5       , 0.        , 0.        ]])




