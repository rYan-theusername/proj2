import os
import sys
import time

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def save_image(img, out_fname, stretch=True):

    if stretch: # stretches pixel values to span all of [0, 255]
        imin, imax = np.min(img), np.max(img)
        img = (img - imin) / (imax - imin) * 255

    # clip to [0, 255] and cast as int
    img = np.clip(img, 0, 255).astype(np.uint8)

    # convert back to PIL and save
    img = Image.fromarray(img)
    img.save(out_fname)


def zero_pad(s, f, mode):

    h_f, w_f = f.shape

    if mode=='valid':
        pad_T, pad_L = 0, 0
        pad_B, pad_R = 0, 0
    if mode=='same':
        pad_T, pad_L = (h_f - 1) // 2, (w_f - 1) // 2
        pad_B, pad_R = h_f - pad_T - 1, w_f - pad_L - 1
    if mode=='full':
        pad_T, pad_L = h_f - 1, w_f - 1
        pad_B, pad_R = pad_T, pad_L

    pad_widths = [(pad_T, pad_B), (pad_L, pad_R)]
    return np.pad(s, pad_width=pad_widths, 
                  mode='constant', constant_values=0.0)


def get_empty_res(s, f, mode):

    h_s, w_s = s.shape
    h_f, w_f = f.shape

    if mode=='valid':
        h_r = h_s - h_f + 1
        w_r = w_s - w_f + 1
    if mode=='same':
        h_r = h_s
        w_r = w_s
    if mode=='full':
        h_r = h_s + h_f - 1
        w_r = w_s + w_f - 1

    return np.empty((h_r, w_r))


def conv_2D_FL(s, f, mode):

    # ---- original dimensions ----
    h_s, w_s = s.shape
    h_f, w_f = f.shape

    # ---- padding ----
    s_padded = zero_pad(s, f, mode)

    # ---- determine result size ----
    res = get_empty_res(s, f, mode)
    h_r, w_r = res.shape

    # ---- flip-and-drag style convolution ----
    ff = np.flip(f)
    for n in range(h_r):
        for m in range(w_r):
            px = 0.0
            for u in range(h_f):
                for v in range(w_f):
                    value = s_padded[n+u, m+v]
                    weight = ff[u, v]
                    px += value * weight
            res[n, m] = px

    return res


def conv_2D_TL(s, f, mode):

    # ---- original dimensions ----
    h_s, w_s = s.shape
    h_f, w_f = f.shape

    # ---- padding ----
    s_padded = zero_pad(s, f, mode)

    # ---- determine result size ----
    res = get_empty_res(s, f, mode)
    h_r, w_r = res.shape
    
    # ---- cookie cutter style convolution ----
    ff = np.flip(f)
    for n in range(h_r):
        for m in range(w_r):
            patch = s_padded[n:n+h_f, m:m+w_f]
            res[n, m] = np.sum(patch * ff)

    return res


def main():

    name = 'self.jpg'
    action = sys.argv[1] if len(sys.argv) > 1 else 'none'
    mode = sys.argv[2] if len(sys.argv) > 2 else 'same'
    implementation = sys.argv[3] if len(sys.argv) > 3 else 'TL'

    # read in image and convert to np array
    in_fname = 'data/' + name
    img = Image.open(in_fname)
    img = np.array(img.convert('L'))
    
    # determine the convolution kernel
    if action == 'blur':
        binomial_1D = np.array(
            [1, 8, 28, 56, 70, 56, 28, 8, 1],
        )
        binomial_2D = np.outer(binomial_1D, binomial_1D)
        filter = binomial_2D / np.sum(binomial_2D)
    elif action == 'boxblur':
        box_1D = np.array(
            [1/81, 1/81, 1/81, 1/81, 1/81, 1/81, 1/81, 1/81, 1/81],
        )
        box_2D = np.outer(box_1D, box_1D)
        filter = box_2D / np.sum(box_2D)
    elif action == 'dx':
        filter = np.array([[1, 0, -1]])
    elif action == 'dy':
        filter = np.array([[1], [0], [-1]])
    else:
        filter = np.array([[1]])

    # apply filter
    if implementation == 'TL':
        img_res = conv_2D_TL(img, filter, mode=mode)
    else:
        img_res = conv_2D_FL(img, filter, mode=mode)

    out_fname = 'results/selfs/' + action + '_' + name
    save_image(img_res, out_fname)


if __name__ == "__main__":
    sys.exit(main())