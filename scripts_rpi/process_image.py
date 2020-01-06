
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
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    img_rot = cv2.warpAffine(img,M,(cols,rows))

    # rotate bounding box
    rect0 = (rect[0], rect[1], 0.0) 
    box = cv2.boxPoints(rect0)
    bcx, bcy = np.mean(box, axis=0, dtype=int)
    [bcx, bcy] = np.int0(cv2.transform(np.array([bcx, bcy]), M))[0]
    pts = np.int0(cv2.transform(np.array([box]), M))[0]
    pts[pts < 0] = 0

    # crop
    img_crop = img_rot[pts[1][1]:pts[0][1], 
                       pts[1][0]:pts[2][0]]

    return img_crop

def find_axe_tip(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(img, kernel, iterations=2)
    _, contours, _ = cv2.findContours(dilated.copy(), 
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = contours[0]
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    axe_cropped = split_rot_image(img, box)

    # center of mass of detected axe
    [vx,vy,cx,cy] = cv2.fitLine(contours, cv2.DIST_L2,0,0.01,0.01)


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
    thresh = cv2.threshold(diff_img, 50, 255,
        cv2.THRESH_BINARY_INV)[1]
    thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    _, contours, _ = cv2.findContours(dilated.copy(), 
        cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    main_cnt = max(contours, key = cv2.contourArea)
    [x, y, w, h] = cv2.boundingRect(main_cnt)
    cv2.rectangle(main_cnt, (x, y), (x + w, y + h), (255, 0, 255), 2)
    print(x, y, w, h)
    rect = cv2.minAreaRect(main_cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    # axe_cropped = split_rot_image(img, box)

    # center of mass of detected axe
    [vx,vy,cx,cy] = cv2.fitLine(main_cnt, cv2.DIST_L2,0,0.01,0.01)
    # find box length along axe
    # axe_tip = find_axe_tip(thresh)

    # # box center
    bcx, bcy = np.mean(box, axis=0, dtype=int)



    cv2.drawContours(thresh_color,[box],0,(255,0,0),2)
    cv2.drawContours(thresh,[box],0,(255,0,0),2)
    cv2.circle(thresh_color, (cx, cy), 10, (0, 0, 255), thickness=3)
    cv2.circle(thresh_color, (bcx, bcy), 10, (0, 255, 0), thickness=3)

    # # 2nd principal axis
    # vx2, vy2 = -vy, -vx
    
    # # https://math.stackexchange.com/questions/274712/calculate-on-which-side-of-a-straight-line-is-a-given-point-located
    # sign = (bcx-cx)*vy2 - (bcy-cy)*vx2  # (𝑥−𝑥1)(𝑦2−𝑦1)−(𝑦−𝑦1)(𝑥2−𝑥1)
    # box_length = calc_box_length(box, vx, vy)
    # print(box_length)
    # if (sign >= 0):

    # else:


    # now just figure out which side is the head
    axe_head = ()

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
