

def get_pad(img, mod_pad):
    # Let's play with the img size
    #img_fn = os.path.join('img/000000.png')
    #print(img_fn)
    #img = cv2.imread(img_fn)
    h,w = img.shape[:2]
    
    MOD = mod_pad
    if w%MOD!=0:
        #pad_w = (8)*(int(w/8) + 1) - w
        pad_w = MOD * (int(w/MOD) + 1) - w
        #print( MOD*(int(w/MOD) + 1) - w)
        if pad_w%2==0:
            pad_w_l, pad_w_r = pad_w/2, pad_w/2 # pad left, right
        else:
            pad_w_l = int(pad_w/2)
            pad_w_r = int(pad_w - pad_w_l)
    else:
        pad_w, pad_w_l, pad_w_r = 0,0,0
    #print('pad_w: %d - pad_w_l: %d - pad_w_l: %d'%(pad_w, pad_w_l, pad_w_r))
    
    if h%MOD!=0:
        pad_h = MOD * (int(h/MOD) + 1) - h
        #print( MOD*(int(h/MOD) + 1) - h)
        if pad_h%2==0:
            pad_h_t, pad_h_b = int(pad_h/2), int(pad_h/2) # pad top, bottom
        else:
            pad_h_t = int(pad_h/2)
            pad_h_b = int(pad_h - pad_h_t)
    else:
        pad_h, pad_h_t, pad_h_b = 0,0,0
    #print('pad_h: %d - pad_h_l: %d - pad_h_l: %d'%(pad_h, pad_h_t, pad_h_b))

    return pad_h_t, pad_h_b, pad_w_l, pad_w_r

