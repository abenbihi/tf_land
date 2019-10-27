

from scipy.misc import logsumexp
import numpy as np
import cv2

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

from tools.cst import *

def resize(img, web_w, web_h):
    """
    Resize img and conserve aspect-ratio. Maybe useless.
    Args:
        img: image
        web_w: width of allocated web canvas
        web_h: height of allocated web canvas
    """
    h,w = img.shape[:2]
    
    # conserve aspect ratio
    s_w = 1.0*web_w/w
    s_h = 1.0*web_h/h
    s = np.minimum(s_w, s_h)
    new_w = int(1.0*w*s)
    new_h = int(1.0*h*s)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img, new_w, new_h


def interpolate_pt(pt, old_w, old_h, new_w, new_h):
    """
    Args:
        pt: opencv convention (x,y)
    """
    x,y = pt
    
    old_min, old_max = 0, old_w
    new_min, new_max = 0, new_w
    new_x = new_min + (x - old_min) * (new_max - new_min) / (old_max - old_min)

    old_min, old_max = 0, old_h
    new_min, new_max = 0, new_h
    new_y = new_min + (y - old_min) * (new_max - new_min) / (old_max - old_min)
    return (int(new_x), int(new_y))


def interpolate_rectangle(x_min, x_max, y_min, y_max, 
        old_w, old_h, new_w, new_h):
    """
    Compute gt waldo for new img size
    Args:
        x_min: x coord. of top-left corner of waldo in [old_h,old_w] img
        x_max: x coord. of bot-right corner of waldo in [old_h,old_w] img
        y_min: y coord. of top-left corner of waldo in [old_h,old_w] img
        y_max: x coord. of bot-right corner of waldo in [old_h,old_w] img
        old_w: width on waldo img on which waldo was found
        old_h: height on waldo img on which waldo was found
        old_w: width of resized waldo img
        old_h: height of resized waldo img
    """
    #GT_W, GT_H = 1500,900 # img size when looking for waldo gt
    DEBUG = (0==1)
    # x
    old_min, old_max = 0, old_w
    new_min, new_max = 0, new_w
    new_x_min = new_min + (x_min - old_min) * (new_max - new_min) / (old_max - old_min)
    new_x_max = new_min + (x_max - old_min) * (new_max - new_min) / (old_max - old_min)

    # y
    old_min, old_max = 0, old_h
    new_min, new_max = 0, new_h
    new_y_min = new_min + (y_min - old_min) * (new_max - new_min) / (old_max - old_min)
    new_y_max = new_min + (y_max - old_min) * (new_max - new_min) / (old_max - old_min)
    
    if DEBUG:
        print('x_min - x_max - y_min - y_max: %d - %d - %d - %d'%(
            x_min, x_max, y_min, y_max))
        print('x_min - x_max - y_min - y_max: %d - %d - %d - %d'%(
            new_x_min, new_x_max, new_y_min, new_y_max))
    return new_x_min, new_x_max, new_y_min, new_y_max


def mask_square_interp(img, square_coord_l):
    """
    First interpolate the square coordinates to the img size then mask the img.
    Args:
        img: of random size
        square_coord_l: (x,y) coordinates of the square starting from top left
        and going trigonometric sens
    """
    (top_l_x, top_l_y, bot_l_x, bot_l_y, bot_r_x, bot_r_y,
            top_r_x, top_r_y) = square_coord_l
    
    old_min, old_max = 0, GT_W
    new_min, new_max = 0, img.shape[1]
    top_l_x = new_min + (top_l_x - old_min) * (new_max - new_min) / (old_max - old_min)
    top_r_x = new_min + (top_r_x - old_min) * (new_max - new_min) / (old_max - old_min)
    bot_l_x = new_min + (bot_l_x - old_min) * (new_max - new_min) / (old_max - old_min)
    bot_r_x = new_min + (bot_r_x - old_min) * (new_max - new_min) / (old_max - old_min)

    # y
    old_min, old_max = 0, GT_H
    new_min, new_max = 0, img.shape[0]
    top_l_y = new_min + (top_l_y - old_min) * (new_max - new_min) / (old_max - old_min)
    top_r_y = new_min + (top_r_y - old_min) * (new_max - new_min) / (old_max - old_min)
    bot_l_y = new_min + (bot_l_y - old_min) * (new_max - new_min) / (old_max - old_min)
    bot_r_y = new_min + (bot_r_y - old_min) * (new_max - new_min) / (old_max - old_min)

    x_min = int(np.minimum(top_l_x, bot_l_x)) 
    x_max = int(np.maximum(top_r_x, bot_r_x))
    y_min = int(np.minimum(top_l_y, top_r_y))
    y_max = int(np.maximum(bot_l_y, bot_r_y))
    #print(x_min, x_max, y_min, y_max)
    img[y_min:y_max, x_min:x_max] = 0
    return img


def mask_square(img, square_coord_l):
    """
    Args:
        img: of size (GT_W, GT_H)
        square_coord_l: (x,y) coordinates of the square starting from top left
        and going trigonometric sens
    """
    (top_l_x, top_l_y, bot_l_x, bot_l_y, bot_r_x, bot_r_y,
            top_r_x, top_r_y) = square_coord_l
    x_min = np.minimum(top_l_x, bot_l_x)
    x_max = np.maximum(top_r_x, bot_r_x)
    y_min = np.minimum(top_l_y, top_r_y)
    y_max = np.maximum(bot_l_y, bot_r_y)
    img[y_min:y_max, x_min:x_max] = 0
    return img

def get_centerbias(size):
    """
    Load centrbias from file, resize it and pre-prroc it for CNN. Output
    shape is [1, size[1], size[0], 1]
    Args: 
        size: (w,h) opencv convention
    """
    centerbias = np.load('meta/centerbias.npy')
    centerbias = cv2.resize(centerbias, size, interpolation=cv2.INTER_CUBIC)
    centerbias -= logsumexp(centerbias)
    centerbias_data = centerbias[np.newaxis, :, :, np.newaxis]
    return centerbias_data


def crop_random(img, crop_size, pt):
    """
    Randomly crop image while keeping point (x,y) in it.
    Args:
        crop_size: (w,h)
        pt: (x,y) position to keep in the crop
    """
    w,h = crop_size
    x,y = pt
    
    print(img.shape[1], img.shape[0])
    print(np.maximum(0, x-w)) 
    print(np.minimum(x, img.shape[1]-w))
    bj = np.random.randint(
            np.maximum(0, x-w), 
            np.minimum(x, img.shape[1]-w))
    ej = bj + w
    bi = np.random.randint(
            np.maximum(0, y-h),
            np.minimum(y, img.shape[0]-h))
    ei = bi + h
    crop = img[bi:ei, bj:ej]
    cv2.imshow('crop', crop)
    cv2.waitKey(0)
    return crop


def crop_middle(img, crop_size, pt):
    """
    Randomly crop image while keeping point (x,y) in it.
    Args:
        crop_size: (w,h)
        pt: (x,y) position to keep in the crop
    """
    w, h = crop_size
    x,y = pt

    bj = np.maximum(
            np.maximum(0, x-int(w/2)), 
            np.minimum(x-int(w/2), img.shape[1]-w))
    ej = bj + w
    bi = np.maximum(
            np.maximum(0, y-int(h/2)), 
            np.minimum(y-int(h/2), img.shape[0]-h))
    ei = bi + h
    #print('bj - ej - bi - ei : %d - %d - %d - %d' %(bj, ej, bi, ei))
    crop = img[bi:ei, bj:ej]
    return crop

def draw_sal_mpl(sal, out_dir, out_fn):
    """
    Draw saliency with matplotlib (as done in offical deep gaze 2)
    Args:
        sal: net output
        out_dir: dir to save the image
        out_fn: prefix to save the img with
    """
    plt.figure(1)
    m = plt.gca().matshow(sal, alpha=0.5, cmap=plt.cm.RdBu)
    plt.colorbar(m)
    plt.title('log density prediction')
    plt.axis('off');
    plt.savefig(os.path.join(out_dir, '%s_log_density.png'%img_root_fn.split(".")[0]))
    plt.close()
    
    plt.figure(1)
    m = plt.gca().matshow(np.exp(sal), alpha=0.5, cmap=plt.cm.RdBu)
    plt.colorbar(m)
    plt.title('density prediction')
    plt.axis('off');
    plt.savefig(os.path.join(out_dir, '%s_density.png'%img_root_fn.split(".")[0]))
    plt.close()


def draw_sal(sal, out_fn):
    heatmap = np.log(-sal + 0.0001)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + .00001)
    heatmap = myjet[np.round(np.clip(heatmap*10, 0, 9)).astype('int'), :]
    heatmap = (heatmap*255).astype('uint8')
    cv2.imwrite(out_fn, heatmap)


