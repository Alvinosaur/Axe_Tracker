#!/usr/bin/env python3

import argparse
import cv2
import numpy as  np
import os
import time

from process_image import count_diff_SSIM
import yaml


verbose = True
YAML_FILEPATH = "default_params.yaml"
CAM_FRAMERATE =  24


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', dest='path', default=YAML_FILEPATH,
                        help='path to yaml file with params')
    parser.add_argument('--img1', dest='img1', required=True, 
                        help='path to first test image')
    parser.add_argument('--img2', dest='img2', required=True,
                        help='path to second test image')
    args = parser.parse_args()
    with open(args.path, 'r') as f:
        params = yaml.load(f)
    img1 = img = cv2.imread(args.img1)
    img1_bgr = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = img = cv2.imread(args.img2)
    img2_bgr = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    width, height = params["image_width"], params["image_height"]
    similarity_threshold = params["similarity_threshold"]
    count_threshold = (params["min_diff_pix"], params["max_diff_pix"])
    # crop_bounds_w = (params["image_min_w"], params["image_max_w"])
    # crop_bounds_h = (params["image_min_h"], params["image_max_h"])
    # already crop image when storing so don't crop again
    crop_bounds_w = (0, width)
    crop_bounds_h = (0, height)

    diff = count_diff_SSIM(img1_bgr, img2_bgr, width, height, 
        crop_bounds_w, crop_bounds_h, similarity_threshold)
    print("min(%d) vs actual(%d) vs max(%d)" % (
        count_threshold[0], diff, count_threshold[1]))
        