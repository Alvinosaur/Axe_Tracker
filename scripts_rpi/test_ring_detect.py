#!/usr/bin/env python3

import argparse
import cv2
import numpy as  np
import os
import time
import yaml
from matplotlib import pyplot as plt


DEBUG = True
YAML_FILEPATH = "default_params.yaml"


class NoRingError(Exception):
    pass

def find_best_ring(img, R_bounds, G_bounds, B_bounds, hough_param1, 
        hough_param2):

    R, G, B = cv2.split(img)
    ret, maskR = cv2.threshold(R, R_bounds[0], R_bounds[1], cv2.THRESH_BINARY_INV)
    ret, maskG = cv2.threshold(G, G_bounds[0], G_bounds[1], cv2.THRESH_BINARY_INV)
    ret, maskB = cv2.threshold(B, B_bounds[0], B_bounds[1], cv2.THRESH_BINARY_INV)

    combined_mask = cv2.bitwise_or(maskR, maskG)
    combined_mask = cv2.bitwise_or(combined_mask, maskB)
    
    if DEBUG:
        # use pyplot instead of cv2 imshow to see actual pixel values
        plt.imshow(combined_mask)
        plt.show()
    
    circles = cv2.HoughCircles(combined_mask, cv2.HOUGH_GRADIENT, dp=1, 
        minDist=1, param1=hough_param1, param2=hough_param2)
        
    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")

        if DEBUG:
            x, y, r = circles[0]
            cv2.circle(img, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            cv2.imshow('img', img)
            cv2.waitKey(0)
        return circles[0]
    
    else:
        return None


def draw_circles(img, circles):
    # draw the circle in the output image, then draw a rectangle
    # corresponding to the center of the circle
    for (x, y, r) in circles:
        cv2.circle(img, (x, y), r, (0, 255, 0), 4)
        cv2.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)


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
    crop_bounds_w = (params["image_min_w"], params["image_max_w"])
    crop_bounds_h = (params["image_min_h"], params["image_max_h"])
    img1 = img1[crop_bounds_h[0]:crop_bounds_h[1], 
                    crop_bounds_w[0]:crop_bounds_w[1]]

    # threshold image by RGB values to isolate rings
    hough_p1, hough_p2 = params["param1"], params["param2"]

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

    # Find rings
    # inner_ring = find_best_ring(img1, R_bounds1, G_bounds1, B_bounds1, 
    #     hough_p1, hough_p2)
    middle_ring = find_best_ring(img1, R_bounds2, G_bounds2, B_bounds2,
        hough_p1, hough_p2)
    # outer_ring = find_best_ring(img1, R_bounds3, G_bounds3, B_bounds3,
    #     hough_p1, hough_p2)

    # if (inner_ring is None) and (middle_ring is None) and (outer_ring is None):
    #     raise(NoRingError("None of the 3 rings were found. Need to re-tune params!"))

    # else:
    #     print(inner_ring)
    #     print(middle_ring)
    #     print(outer_ring)
    #     rings = fill_missing_rings(inner_ring, middle_ring, outer_ring)
    #     draw_circles(img1, rings)
    #     cv2.imshow("img1", img1)
    #     cv2.waitKey(0)
    #     return rings


def fill_missing_rings(ring1, ring2, ring3):
    out_to_in = 30  # inner radius + 30 = outer radius
    out_to_mid = 10
    mid_to_in = 15
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

    return ring1, ring2, ring3


if __name__ == '__main__':
    main()