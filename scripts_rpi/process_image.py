
import skimage as ski  # scikit-image (for processing images)
import imutils
import cv2		# opencv library
import numpy as np
import matplotlib as mp
mp.use('TkAgg')	# tkinter (for displaying images)
from matplotlib import pyplot as plt

verbose = True

def count_diff_SSIM(img1, img2, width, height, crop_bounds_w, crop_bounds_h, 
        similarity_threshold):
    # crop image to only show target
    raw_img1 = img1[crop_bounds_h[0]:crop_bounds_h[1], 
                    crop_bounds_w[0]:crop_bounds_w[1]]
    raw_img2 = img2[crop_bounds_h[0]:crop_bounds_h[1], 
                    crop_bounds_w[0]:crop_bounds_w[1]]

    # convert the images to grayscale
    gray_img1 = cv2.cvtColor(raw_img1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(raw_img2, cv2.COLOR_BGR2GRAY)

    # compute Structural Similarity Index of the two images
    (score, diff_img) = ski.measure.compare_ssim(gray_img1, 
        gray_img2, full=True)
    diff_img = (diff_img * 255).astype("uint8")

    # threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff_img, similarity_threshold, 1, 
        cv2.THRESH_BINARY_INV)[1]
    if verbose:
        fig = plt.figure()
        print(np.sum(thresh))
        cv2.imshow("New Image", raw_img1)
        cv2.imshow("Prev Image", raw_img2)
        plt.imshow(thresh)
        plt.show()
        cv2.waitKey(1)
    return np.sum(thresh)
