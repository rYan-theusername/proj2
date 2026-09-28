import os
import sys
import time

from PIL import Image
import numpy as np
import scipy
import cv2

import matplotlib.pyplot as plt

from homemade_convolution import save_image


def conv_channels(img, filter):

    if img.ndim == 2:
        return scipy.signal.fftconvolve(img, filter, mode='same')
        
    convolved_channels = [
        scipy.signal.fftconvolve(img[:, :, i], filter, mode='same')
        for i in range(3)
    ]
    return np.stack(convolved_channels, axis=2)


def save_or_display(img, fname, save):
    
    if save == 'True':      # convert back to an image
        save_image(img, fname, stretch=False)

    else:                   # display all images
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.imshow(img)
        ax.set_title(fname)
        ax.axis('off')
        plt.tight_layout()
        plt.show()


def main():

    name = sys.argv[1] if len(sys.argv) > 1 else 'cameraman.jpg'
    blur_level = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    sharp_level = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    save = sys.argv[4] if len(sys.argv) > 4 else 'False'
    blur_then_sharp = sys.argv[5] if len(sys.argv) > 5 else 'False'
    viz_high_pass = sys.argv[6] if len(sys.argv) > 6 else 'False'

    # read in image and convert to np array
    in_fname = 'data/' + name
    img = Image.open(in_fname)
    img = np.array(img)
    
    # define the gaussian blur kernel
    gaussian1D = cv2.getGaussianKernel(blur_level, 0)
    gaussian2D = np.outer(gaussian1D, gaussian1D)
    gaussian2D = gaussian2D / np.sum(gaussian2D)

    # define the identity convolution operator
    I = np.zeros_like(gaussian2D)
    I[I.shape[0] // 2, I.shape[0] // 2] = 1

    # subtracting these gives a HPF
    high_pass = I - gaussian2D

    # mix the HPF and the identity for a sharpener
    sharp_filter = I + sharp_level*high_pass

    # apply the filter
    sharpened_img = conv_channels(img, sharp_filter)

    fname = 'results/sharp_mask/' + str(blur_level) + '_' + str(sharp_level) + '_' + name
    sharpened_img = np.clip(sharpened_img, 0, 255).astype(np.uint8)
    save_or_display(sharpened_img, fname, save)

    if blur_then_sharp == 'True':
        blurred_img = conv_channels(img, gaussian2D)
        sharpened_blurred_img = conv_channels(blurred_img, sharp_filter)
        sharpened_blurred_img = np.clip(sharpened_blurred_img, 0, 255).astype(np.uint8)
        fname = 'results/sharp_mask/blurredfirst_' + str(blur_level) + '_' + str(sharp_level) + '_' + name
        save_or_display(sharpened_blurred_img, fname, save)

    if viz_high_pass == 'True':
        high_passed_img = conv_channels(img, high_pass)
        peak = np.max(np.abs(high_passed_img))
        high_passed_img = 128 + high_passed_img / peak * 127
        high_passed_img = np.clip(high_passed_img, 0, 255).astype(np.uint8)
        fname = 'results/sharp_mask/high_pass_' + str(blur_level) + '_' + name
        save_or_display(high_passed_img, fname, save)


if __name__ == "__main__":
    sys.exit(main())