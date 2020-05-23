#!/usr/bin/env python3

import argparse
import cv2
import numpy as  np
import os
import time
import yaml
from matplotlib import pyplot as plt

from helpers.ransac_circle import run_ransac

DEBUG = False
MASK_VAL = 255
YAML_FILEPATH = "params/default_params.yaml"


class NoRingError(Exception):
    pass

def find_best_ring(img, R_bounds, G_bounds, B_bounds, hough_param1, 
        hough_param2, hough_dp, ring_est_dist):

    R, G, B = cv2.split(img)
    ret, maskR_low = cv2.threshold(R, R_bounds[0], MASK_VAL, cv2.THRESH_BINARY)
    ret, maskR_high = cv2.threshold(R, R_bounds[1], MASK_VAL, cv2.THRESH_BINARY_INV)
    maskR = cv2.bitwise_and(maskR_low, maskR_high)

    ret, maskG_low = cv2.threshold(G, G_bounds[0], MASK_VAL, cv2.THRESH_BINARY)
    ret, maskG_high = cv2.threshold(G, G_bounds[1], MASK_VAL, cv2.THRESH_BINARY_INV)
    maskG = cv2.bitwise_and(maskG_low, maskG_high)

    ret, maskB_low = cv2.threshold(B, B_bounds[0], MASK_VAL, cv2.THRESH_BINARY)
    ret, maskB_high = cv2.threshold(B, B_bounds[1], MASK_VAL, cv2.THRESH_BINARY_INV)
    maskB = cv2.bitwise_and(maskB_low, maskB_high)

    combined_mask = cv2.bitwise_and(maskR, maskG)
    combined_mask = cv2.bitwise_and(combined_mask, maskB)
    
    if DEBUG:
        # use pyplot instead of cv2 imshow to sece actual pixel values
        plt.imshow(combined_mask)
        plt.show()
    
    # circles = cv2.HoughCircles(combined_mask, cv2.HOUGH_GRADIENT, dp=hough_dp, 
    #     minDist=1, param1=hough_param1, param2=hough_param2)
    pts = np.transpose(np.nonzero(combined_mask))
    (ransac_center, ransac_rad), _ = run_ransac(pts, max_iters=2000, 
        inlier_thresh=0.05*ring_est_dist)
    circles = [(int(ransac_center[0]), int(ransac_center[1]), int(ransac_rad))]
        
    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        # circles = np.round(circles[0, :]).astype("int")

        if DEBUG:
            x, y, r = circles[0]
            output = img.copy()
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            cv2.imshow('img', output)
            cv2.waitKey(0)
        return circles[0]
    
    else:
        return None


def draw_circles(img, circles, thickness=3):
    # draw the circle in the output image, then draw a rectangle
    # corresponding to the center of the circle
    for (x, y, r) in circles:
        cv2.circle(img, (x, y), r, (0, 255, 0), thickness)
        cv2.rectangle(img, (x - 2, y - 2), (x + 2, y + 2), (0, 128, 255), -1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', dest='path', default=YAML_FILEPATH,
                        help='path to yaml file with params')
    parser.add_argument('--img1', dest='img1', required=True, 
                        help='path to first test image')
    args = parser.parse_args()
    with open(args.path, 'r') as f:
        params = yaml.load(f)
    img1 = cv2.imread(args.img1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    
    # crop image to focus on the target
    # crop_bounds_w = (params["image_min_w"], params["image_max_w"])
    # crop_bounds_h = (params["image_min_h"], params["image_max_h"])
    # img1 = img1[crop_bounds_h[0]:crop_bounds_h[1], 
    #                 crop_bounds_w[0]:crop_bounds_w[1]]

    # threshold image by RGB values to isolate rings
    hough_p1, hough_p2 = params["param1"], params["param2"]
    hough_dp = params["hough_dp"]

    # Inner ring
    R_bounds1 = (params["R_min1"], params["R_max1"])
    G_bounds1 = (params["G_min1"], params["G_max1"])
    B_bounds1 = (params["B_min1"], params["B_max1"])

    # Middle ring
    R_bounds2 = (params["R_min2"], params["R_max2"])
    G_bounds2 = (params["G_min2"], params["G_max2"])
    B_bounds2 = (params["B_min2"], params["B_max2"])

    # Outer ring
    R_bounds3 = (params["R_min3"], params["R_max3"])
    G_bounds3 = (params["G_min3"], params["G_max3"])
    B_bounds3 = (params["B_min3"], params["B_max3"])

    # Ring distances
    is_mini_img = True
    if is_mini_img:
        ring_est_dist = params["ring_est_dist_mini"]
    else:
        ring_est_dist = params["ring_est_dist"]

    # Find rings
    inner_ring = find_best_ring(img1, R_bounds1, G_bounds1, B_bounds1, 
        hough_p1, hough_p2, hough_dp, ring_est_dist)
    middle_ring = find_best_ring(img1, R_bounds2, G_bounds2, B_bounds2,
        hough_p1, hough_p2, hough_dp, 2*ring_est_dist)
    outer_ring = find_best_ring(img1, R_bounds3, G_bounds3, B_bounds3,
        hough_p1, hough_p2, hough_dp, 3*ring_est_dist)

    middle_ring = None
    inner_ring = None

    if (inner_ring is None) and (middle_ring is None) and (outer_ring is None):
        raise(NoRingError("None of the 3 rings were found. Need to re-tune params!"))

    else:
        rings = fill_missing_rings(inner_ring, middle_ring, outer_ring, 
            out_to_mid=ring_est_dist, mid_to_in=ring_est_dist)
        draw_circles(img1, rings, thickness=2)
        cv2.imshow("img1", img1)
        cv2.waitKey(0)
        return rings


def fill_missing_rings(ring1, ring2, ring3, out_to_mid=65, mid_to_in=65):
    # returns original rings if all rings are present
    # else approximates missing rings as long as one ring present
    out_to_in = out_to_mid + mid_to_in
    if ring1 is None:
        if ring2 is None:
            ring1 = (ring3[0], ring3[1], ring3[2]-out_to_in)
            ring2 = (ring3[0], ring3[1], ring3[2]-out_to_mid)
        else:
            ring1 = (ring2[0], ring2[1], ring2[2]-mid_to_in)
            if ring3 is None:
                ring3 = (ring2[0], ring2[1], ring2[2]+out_to_mid)
    elif ring2 is None:
        if ring3 is None:
            ring2 = (ring1[0], ring1[1], ring1[2]+mid_to_in)
            ring3 = (ring1[0], ring1[1], ring1[2]+out_to_in)
        else:
            # r1 = (r2/r3)*r2, assuming each level has same scale down
            ring2 = (ring3[0], ring3[1], ring3[2]-out_to_mid)
            if ring1 is None:
                ring1 = (ring3[0], ring3[1], ring3[2]-out_to_in)
    elif ring3 is None:
        if ring1 is None:
            ring1 = (ring2[0], ring2[1], ring2[2]-mid_to_in)
            ring3 = (ring2[0], ring2[1], ring2[2]+out_to_mid)
        else:
            # r1 = (r2/r3)*r2, assuming each level has same scale down
            ring3 = (ring1[0], ring1[1], ring1[2]+out_to_in)
            if ring2 is None:
                ring2 = (ring1[0], ring1[1], ring1[2]+mid_to_in)

    return [ring1, ring2, ring3]


if __name__ == '__main__':
    main()