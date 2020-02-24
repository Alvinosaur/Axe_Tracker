
import skimage as ski  # scikit-image (for processing images)
import imutils
import cv2		# opencv library
import numpy as np
import matplotlib as mp
mp.use('TkAgg')	# tkinter (for displaying images)
from matplotlib import pyplot as plt

verbose = False

def split_rot_image(img, rect):
    # rotate img
    angle = rect[2]
    rows, cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols//2,rows//2),angle,1)
    img_rot = cv2.warpAffine(img,M,(cols,rows))

    return img_rot

def find_axe_tip(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(img, kernel, iterations=2)
    _, contours, _ = cv2.findContours(dilated.copy(), 
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    main_cnt = max(contours, key = cv2.contourArea)
    [x, y, w, h] = cv2.boundingRect(main_cnt)
    cv2.rectangle(main_cnt, (x, y), (x + w, y + h), (255, 0, 255), 2)
    rect = cv2.minAreaRect(main_cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    axe_cropped = split_rot_image(img[y:y+h, x:x+w], rect)
    first_half, second_half = axe_cropped[:h//2, :], axe_cropped[h//2:h, :]  

    if np.sum(first_half) > np.sum(second_half):
        return (x+w, y+h//2), box
    else:
        return (x, y+h//2), box


def count_diff_SSIM(img1, img2, similarity_threshold):
    # convert the images to grayscale
    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # compute Structural Similarity Index of the two images
    (score, diff_img) = ski.measure.compare_ssim(gray_img1, 
        gray_img2, full=True)
    diff_img = (diff_img * 255).astype("uint8")

    # threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff_img, similarity_threshold, 255,
        cv2.THRESH_BINARY_INV)[1]
    thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    axe_tip, box = find_axe_tip(thresh)

    cv2.drawContours(thresh_color,[box],0,(255,0,0),2)
    cv2.circle(thresh_color, axe_tip, 10, (0, 255, 0), thickness=3)

    if verbose:
        fig = plt.figure()
        cv2.imshow("New Image", img1)
        cv2.imshow("Prev Image", img2)
        cv2.imshow("Differences", thresh_color)
        plt.imshow(thresh_color)
        plt.show()
        cv2.waitKey(0)

    total = 0
    cx, cy = 0, 0
    rows, cols = thresh.shape
    return np.sum(thresh), thresh, thresh_color
