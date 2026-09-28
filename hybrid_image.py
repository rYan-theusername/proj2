import math
import os
import sys
import time

import numpy as np
import scipy
import cv2

from PIL import Image

from align_image_code import align_images
from image_sharpening import conv_channels, save_or_display


def make_LPF(f_cutoff):

    sigma = 1 / (2*np.pi*f_cutoff)    # convert cycles/pixel cutoff to gaussian standard deviation
    num_taps = 2*math.ceil(3*sigma) + 1     # use >3 standard deviations of taps on either side

    gaussian1D = cv2.getGaussianKernel(num_taps, sigma)
    gaussian2D = np.outer(gaussian1D, gaussian1D)

    return gaussian2D / np.sum(gaussian2D)


def make_HPF(sigma2):

    LPF = make_LPF(sigma2)
    I = np.zeros_like(LPF)
    I[I.shape[0] // 2, I.shape[0] // 2] = 1
    return I - LPF


def fft_viz(img, fname, save):

    if img.ndim == 3:   # conversion to grayscale
        img = 0.299*img[:, :, 0] + 0.587*img[:, :, 1] + 0.114*img[:, :, 2]

    im_FT = np.log(np.abs(np.fft.fftshift(np.fft.fft2(img))) + 1e-8)
    im_FT = (im_FT - im_FT.min()) / (im_FT.max() - im_FT.min() + 1e-12) * 255
    save_or_display(im_FT, fname, save)


def main():

    name = sys.argv[1] if len(sys.argv) > 1 else 'derek_and_nutmeg'
    f_hp = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
    f_lp = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01
    hp_gain = float(sys.argv[4]) if len(sys.argv) > 4 else 2
    save = sys.argv[5] if len(sys.argv) > 5 else 'False'
    do_fft = sys.argv[6] if len(sys.argv) > 6 else 'False'

    if name == 'derek_and_nutmeg':
        im1_fname = 'data/nutmeg.jpg'
        im2_fname = 'data/derek.jpg'
    elif name == 'anakin_and_vader':
        im1_fname = 'data/anakin.jpg'
        im2_fname = 'data/vader.jpg'
    elif name == 'whitehouse':
        im1_fname = 'data/wh_before.jpg'
        im2_fname = 'data/wh_after.jpg'

    im1 = np.array(Image.open(im1_fname))
    im2 = np.array(Image.open(im2_fname))

    im1_hp = conv_channels(im1, make_HPF(f_hp))
    im2_lp = conv_channels(im2, make_LPF(f_lp))

    im1_hp_aligned, im2_lp_aligned = align_images(im1_hp, im1, im2_lp, im2)
    hybrid = hp_gain*im1_hp_aligned + im2_lp_aligned

    fname = 'results/hybrid/' + name + '.jpg'
    save_or_display(np.clip(hybrid, 0, 255).astype(np.uint8), fname, save)

    if do_fft == 'True':
        fft_viz(im1_hp, 'results/hybrid/fft_hp_' + os.path.basename(im1_fname), save)
        fft_viz(im2_lp, 'results/hybrid/fft_lp_' + os.path.basename(im2_fname), save)
        fft_viz(im1, 'results/hybrid/fft_' + os.path.basename(im1_fname), save)
        fft_viz(im2, 'results/hybrid/fft_' + os.path.basename(im2_fname), save)
        fft_viz(hybrid, 'results/hybrid/fft_hybrid_' + name + '.jpg', save)


if __name__ == "__main__":
    sys.exit(main())
