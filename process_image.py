verbose = True

# import libraries
if verbose:
    print('Loading process_image modules...')
import time
#import picamera     	# Raspberry Pi camera
import numpy as np
import matplotlib as mp
mp.use('TkAgg')	# tkinter (for displaying images)
from matplotlib import pyplot as plt
import skimage as ski  # scikit-image (for processing images)
import imutils
import cv2		# opencv library

# import functions from grab_image.py file
#from grab_image import *

IMG_EMPTY = 'axe_images/noAxe.png'      # sample image of empty target
IMG_AXE1 = 'axe_images/real1.png'       # sample image with axe on target
IMG_AXE2 = 'axe_images/score1_4.png'    # sample image with taped axe on target
IMG_BLOCK = 'axe_images/blocked2.png'    # sample image with blockage

def display_diff_SSIM(img1, img2):
    #  width, height, _  =  cam.resolution()
    # to display several figures at once
    fig = plt.figure()

    # cropped raw BGR images: square with x in [100, 200], y in [40, 140]
    raw_img1 = cv2.imread(img1)
    raw_img2 = cv2.imread(img2)
    width, height, _ = raw_img1.shape
    inc_w = width // 4
    inc_h = height // 4
    raw_img1 = raw_img1[inc_w:3*inc_w, inc_h:3*inc_h]
    raw_img2 = raw_img2[inc_w:3*inc_w, inc_h:3*inc_h]
    cv2.imshow("orig1",  raw_img1)
    cv2.imshow("orig2",  raw_img2)

    # convert the images to grayscale
    gray_img1 = cv2.cvtColor(raw_img1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(raw_img2, cv2.COLOR_BGR2GRAY)

    # compute Structural Similarity Index of the two images
    (score, diff_img) = ski.measure.compare_ssim(gray_img1, 
        gray_img2, full=True)
    diff_img = (diff_img * 255).astype("uint8")
    # gray_diff_img = cv2.cvtColor(diff_img, cv2.COLOR_BGR2GRAY)

    # threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff_img, 10, 1, cv2.THRESH_BINARY_INV)[1]
    print(np.sum(thresh))
    plt.imshow(thresh)
    plt.show()

# TESTING ZONE

display_diff_SSIM(IMG_EMPTY, IMG_AXE1)
# display_diff_SSIM("scripts_rpi/2019-09-26_13:02:42.png", 
#     "scripts_rpi/2019-09-26_13:02:49.png")
