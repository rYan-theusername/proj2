import os
import sys
import time

from PIL import Image
import numpy as np
import scipy
import cv2

import matplotlib.pyplot as plt

from homemade_convolution import save_image
from hybrid_image import make_LPF


def conv_channels_reflect(img, filter):

    pad = filter.shape[0] // 2

    if img.ndim == 2:
        padded = np.pad(img, pad, mode='reflect')
        return scipy.signal.fftconvolve(padded, filter, mode='valid')
    
    padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    convolved_channels = [
        scipy.signal.fftconvolve(padded[:, :, i], filter, mode='valid')
        for i in range(3)
    ]
    return np.stack(convolved_channels, axis=2)


def gaussian_stack(img, levels=4, scale=1/2):
    # levels = how many images are in the stack (+ 1 for the original)
    # scale = how quickly does the frequency cutoffs decrease (closer to 0 for faster)
    results = [img]
    for i in range(levels):
        kernel = make_LPF(0.5 * (scale)**i)
        results.append(conv_channels_reflect(img, kernel))
    
    return results


def laplacian_stack(g_stack):

    results = []
    for i in range(len(g_stack) - 1):
        results.append(g_stack[i] - g_stack[i+1])
    results.append(g_stack[-1])

    return results


def main():

    name = sys.argv[1] if len(sys.argv) > 1 else 'apple.jpeg'
    stack_level = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    save = sys.argv[3] if len(sys.argv) > 3 else 'False'

    # read in image and convert to np array
    in_fname = 'data/' + name
    img = Image.open(in_fname)
    img = np.array(img)
    
    g_stack = gaussian_stack(img, levels=stack_level)

    # make filenames and clip the results to [0, 255]
    labels, imgs = [], []
    for i in range(len(g_stack)):
        labels.append('results/gaussian_stacks/level' + str(i) + '_' + os.path.basename(in_fname))
        imgs.append(np.clip(g_stack[i], 0, 255).astype(np.uint8))
        
    if save == 'True':      # convert back to an image
        for img, fname in zip(imgs, labels):
            save_image(img, fname)

    else:                   # display all images
        fig, axes = plt.subplots(len(imgs), 1, figsize=(4, len(imgs)*4))
        for img, label, ax in zip(imgs, labels, axes):
            ax.imshow(img)
            ax.set_title(label)
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    l_stack = laplacian_stack(g_stack)

    # make filenames and clip the results to [0, 255]
    labels, imgs = [], []
    for i, l in enumerate(l_stack):
        labels.append('results/laplacian_stacks/level' + str(i) + '_' + os.path.basename(in_fname))
        if i < len(l_stack) - 1:    # rescale all but final level of the stack
            peak = np.max(np.abs(l))
            l = 128 + l / peak * 127
        imgs.append(np.clip(l, 0, 255).astype(np.uint8))
        
    if save == 'True':      # convert back to an image
        for img, fname in zip(imgs, labels):
            save_image(img, fname, stretch=False)

    else:                   # display all images
        fig, axes = plt.subplots(len(imgs), 1, figsize=(4, len(imgs)*4))
        for img, label, ax in zip(imgs, labels, axes):
            ax.imshow(img)
            ax.set_title(label)
            ax.axis('off')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    sys.exit(main())