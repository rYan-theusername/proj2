import os
import sys
import time

from PIL import Image
import numpy as np
import scipy
import cv2

import matplotlib.pyplot as plt

from homemade_convolution import save_image
from stacks import gaussian_stack, laplacian_stack


def main():

    stack_level = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    save = sys.argv[2] if len(sys.argv) > 2 else 'False'
    oraple = sys.argv[3] if len(sys.argv) > 3 else 'oraple'

    # read in image and convert to np array
    if oraple == 'oraple':
        app = Image.open('data/apple.jpeg')
        app = np.array(app)
        orr = Image.open('data/orange.jpeg')
        orr = np.array(orr)
    if oraple == 'nyc':
        app = Image.open('data/nyc_sat.jpg')
        app = np.array(app)
        orr = Image.open('data/nyc_map.jpg')
        orr = np.array(orr)
    if oraple == 'whitehouse':
        app = Image.open('data/wh_before.jpg')
        app = np.array(app)
        orr = Image.open('data/wh_after.jpg')
        orr = np.array(orr)

    mask = np.zeros_like(orr)
    if oraple == 'oraple':
        mask[:, :mask.shape[1]//2] = 1.0
        scale = 1/2
    if oraple == 'nyc':
        mask[:, :mask.shape[1]//2] = 1.0
        scale = 1/2
    if oraple == 'whitehouse':
        mask[:mask.shape[1]//2, :] = 1.0
        scale = 1/2

    masks = gaussian_stack(mask, levels=stack_level, scale=scale)
    apps = laplacian_stack(gaussian_stack(app, levels=stack_level))
    orrs = laplacian_stack(gaussian_stack(orr, levels=stack_level))

    blended = []
    for i in range(stack_level+1):
        blended.append(apps[i] * masks[i] + orrs[i] * (1 - masks[i]))

    result = np.zeros_like(orr, dtype=np.float64)
    for blended_lvl in blended:
        result += blended_lvl

    # make filenames and clip the results to [0, 255]
    labels, imgs = [], []
    for i in range(len(blended)):
        left = apps[i] * masks[i]
        right = orrs[i] * (1 - masks[i])
        for name, level in [('L', left), ('R', right), ('blended', blended[i])]:
            labels.append('results/' + oraple + '/level' + str(i) + '_' + name + '.jpg')
            if i < len(blended) - 1:    # rescale all but final level of the stack
                peak = np.max(np.abs(level))
                level = 128 + level / peak * 127
            imgs.append(np.clip(level, 0, 255).astype(np.uint8))

    labels.append('results/' + oraple + '/' + oraple + '.jpg')
    imgs.append(np.clip(result, 0, 255).astype(np.uint8))

    if save == 'True':      # convert back to an image
        for img, fname in zip(imgs, labels):
            save_image(img, fname, stretch=False)

    else:                   # display all images
        fig, axes = plt.subplots(len(blended), 3, figsize=(12, len(blended)*4))
        for img, label, ax in zip(imgs[:-1], labels[:-1], axes.ravel()):
            ax.imshow(img)
            ax.set_title(label)
            ax.axis('off')
        plt.tight_layout()
        plt.show()

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.imshow(imgs[-1])
        ax.set_title(labels[-1])
        ax.axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    sys.exit(main())