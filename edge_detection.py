import os
import sys
import time

from PIL import Image
import numpy as np
import scipy
import cv2

import matplotlib.pyplot as plt

from homemade_convolution import save_image, zero_pad


def main():

    name = 'cameraman.jpg'
    blur_level = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    threshold_level = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
    dog = sys.argv[3] if len(sys.argv) > 3 else 'False'
    save = sys.argv[4] if len(sys.argv) > 4 else 'False'

    # read in image and convert to np array
    in_fname = 'data/' + name
    img = Image.open(in_fname)
    img = np.array(img.convert('L'))
    
    # define the gaussian blur kernel
    gaussian1D = cv2.getGaussianKernel(blur_level, 0)
    gaussian2D = np.outer(gaussian1D, gaussian1D)
    gaussian2D = gaussian2D / np.sum(gaussian2D)

    # define the difference / blurred difference kernels
    dx = np.array([[1, 0, -1]])
    dx_b = scipy.signal.convolve2d(dx, gaussian2D, mode='full')
    dy = np.array([[1], [0], [-1]])
    dy_b = scipy.signal.convolve2d(dy, gaussian2D, mode='full')

    laplacian = np.array([
        [0,  1, 0],
        [1, -4, 1],
        [0,  1, 0]
    ])
    laplacian_b = scipy.signal.convolve2d(laplacian, gaussian2D, mode='full')
    save_image(laplacian, 'results/filters/laplacian.jpg')

    if dog=='True':     # easy option: use the blurred filters
        img_dx = scipy.signal.convolve2d(img, dx_b, mode='same')
        img_dy = scipy.signal.convolve2d(img, dy_b, mode='same')
        img_grad = np.sqrt(img_dx**2 + img_dy**2)
        img_laplacian = scipy.signal.convolve2d(img, laplacian_b, mode='same')

    else:               # hard option: filter the blurred image
        img_bdx = zero_pad(img, dx_b, mode='same')
        img_bdx = scipy.signal.convolve2d(img_bdx, gaussian2D, mode='valid')
        img_dx = scipy.signal.convolve2d(img_bdx, dx, mode='valid')

        img_bdy = zero_pad(img, dy_b, mode='same')
        img_bdy = scipy.signal.convolve2d(img_bdy, gaussian2D, mode='valid')
        img_dy = scipy.signal.convolve2d(img_bdy, dy, mode='valid')

        img_grad = np.sqrt(img_dx**2 + img_dy**2)

        img_blaplacian = zero_pad(img, laplacian_b, mode='same')
        img_blaplacian = scipy.signal.convolve2d(img_blaplacian, gaussian2D, mode='valid')
        img_laplacian = scipy.signal.convolve2d(img_blaplacian, laplacian, mode='valid')

    grad_threshold = np.where(np.abs(img_grad) < threshold_level*np.max(np.abs(img_grad)), 0, img_grad)
    laplacian_threshold = np.where(np.abs(img_laplacian) < threshold_level*np.max(np.abs(img_laplacian)), 0, img_laplacian)

    images = [img_dx, img_dy, img_grad, img_laplacian,
              grad_threshold, laplacian_threshold]
    labels = ['dx_' + name, 
            'dy_' + name,
            'grad_' + name, 
            'laplacian_' + name,
            'grad_thresholded_' + str(threshold_level) + '_' + name,
            'laplacian_thresholded_' + str(threshold_level) + '_' + name
            ]

    if save == 'True':      # convert back to an image
        for img, label in zip(images, labels):
            if dog=='True':
                fname = 'results/blurred_filters_' + str(blur_level) + '/' + label
            else:
                fname = 'results/blurred_imgs_' + str(blur_level) + '/' + label
            save_image(img, fname)

    else:                   # display all images
        fig, axes = plt.subplots(3, 2, figsize=(8, 12))

        for i, ax in enumerate(axes.flat):
            ax.imshow(images[i], cmap='gray')
            ax.axis('off')
            ax.set_title(labels[i])

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    sys.exit(main())