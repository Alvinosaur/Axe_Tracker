
import skimage as ski  # scikit-image (for processing images)
import imutils
import cv2		# opencv library
import numpy as np
import matplotlib as mp
mp.use('TkAgg')	# tkinter (for displaying images)
from matplotlib import pyplot as plt

verbose = True

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
    thresh = cv2.threshold(diff_img, similarity_threshold, 255,
        cv2.THRESH_BINARY_INV)[1]
    thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    axe_tip, box = find_axe_tip(thresh)

    cv2.drawContours(thresh_color,[box],0,(255,0,0),2)
    cv2.circle(thresh_color, axe_tip, 10, (0, 255, 0), thickness=3)

    rings = find_rings()
    score = calc_score()

    if verbose:
        fig = plt.figure()
        cv2.imshow("New Image", raw_img1)
        cv2.imshow("Prev Image", raw_img2)
        cv2.imshow("Differences", thresh_color)
        plt.imshow(thresh_color)
        plt.show()
        cv2.waitKey(1)

    total = 0
    cx, cy = 0, 0
    rows, cols = thresh.shape
    for r in range(rows):
        for c in range(cols):
            if thresh[r][c] == 1: 
                cx += c
                cy += r
                total += 1
    cx /= total
    cy /= total
    print(cx, cy)
    return np.sum(thresh), thresh


# def find_rings():



# def calc_score():
#     # for each ring, check if axe tip lies in ring (dist from origin <= radius with some error threshold)
#     # start from tightest ring, work way outside, if find match, report this as best score

#     return score