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


def black_mask(img):
    # 1 where the mask image is black, 0 on the white background
    if img.ndim == 3:
        black = np.mean(img, axis=2) < 128
    else:
        black = img < 128
    mask = np.zeros(img.shape[:2], dtype=np.float64)
    mask[black] = 1
    return mask


def main():

    save = sys.argv[1] if len(sys.argv) > 1 else 'False'
    name = sys.argv[2] if len(sys.argv) > 2 else 'swamp'
    stack_level = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    scale = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5

    # read in image and convert to np array
    if name == 'swamp':
        obj = Image.open('data/swamp_obj.jpg')
        obj = np.array(obj)
        bkg = Image.open('data/swamp_bkg.jpg')
        bkg = np.array(bkg)
        msk = Image.open('data/swamp_msk.jpg')
        msk = np.array(msk)
    if name == 'jeffryan':
        obj = Image.open('data/jeffryan_obj.jpg')
        obj = np.array(obj)
        bkg = Image.open('data/jeffryan_bkg.jpg')
        bkg = np.array(bkg)
        msk = Image.open('data/jeffryan_msk.jpg')
        msk = np.array(msk)
    if name == 'yung_arvind':
        obj = Image.open('data/yung_arvind_obj.jpg')
        obj = np.array(obj)
        bkg = Image.open('data/yung_arvind_bkg.jpg')
        bkg = np.array(bkg)
        msk = Image.open('data/yung_arvind_msk.jpg')
        msk = np.array(msk)


    mask = black_mask(msk)
    obj = obj.astype(np.float64)
    bkg = bkg.astype(np.float64)

    masks = gaussian_stack(mask, levels=stack_level, scale=scale)
    masks = [mask[:, :, None] for mask in masks]    # add axis for broadcasting
    objs = laplacian_stack(gaussian_stack(obj, levels=stack_level))
    bkgs = laplacian_stack(gaussian_stack(bkg, levels=stack_level))

    blended = []
    for i in range(stack_level+1):
        blended.append(objs[i] * masks[i] + bkgs[i] * (1 - masks[i]))

    result = np.zeros_like(bkg, dtype=np.float64)
    for blended_lvl in blended:
        result += blended_lvl

    # make filenames and clip the results to [0, 255]
    labels, imgs = [], []
    for i in range(len(blended)):
        left = objs[i] * masks[i]
        right = bkgs[i] * (1 - masks[i])
        for side, level in [('L', left), ('R', right), ('blended', blended[i])]:
            labels.append('results/' + name + '/level' + str(i) + '_' + side + '.jpg')
            if i < len(blended) - 1:    # rescale all but final level of the stack
                peak = np.max(np.abs(level))
                level = 128 + level / peak * 127
            imgs.append(np.clip(level, 0, 255).astype(np.uint8))

    labels.append('results/' + name + '/' + name + '.jpg')
    imgs.append(np.clip(result, 0, 255).astype(np.uint8))

    if save == 'True':      # convert back to an image
        os.makedirs('results/' + name, exist_ok=True)
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
