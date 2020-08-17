#!/usr/bin/env python3

import argparse
import cv2
import numpy as np
import os
import time

from classical_approach.process_image import count_diff_SSIM
import yaml


verbose = True
YAML_FILEPATH = "params/default_params.yaml"
CAM_FRAMERATE = 24


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

    width, height = params["image_width"], params["image_height"]
    similarity_threshold = params["similarity_threshold"]
    count_threshold = (params["min_diff_pix"], params["max_diff_pix"])
    crop_bounds_w = (params["image_min_w"], params["image_max_w"])
    crop_bounds_h = (params["image_min_h"], params["image_max_h"])

    num_iters = 100
    avg_time = 0
    for iter in range(num_iters):
        start = time.time()

        img1 = cv2.cvtColor(cv2.imread(args.img1), cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(cv2.imread(args.img2), cv2.COLOR_BGR2RGB)
        cropped_img1 = img1[:, :]
        cropped_img2 = img2[:, :]
        # cropped_img1 = img1[crop_bounds_h[0]:crop_bounds_h[1],
        #                     crop_bounds_w[0]:crop_bounds_w[1]]
        # cropped_img2 = img2[crop_bounds_h[0]:crop_bounds_h[1],
        #                     crop_bounds_w[0]: crop_bounds_w[1]]
        # already crop image when storing so don't crop again
        # crop_bounds_w = (600, width)
        # crop_bounds_h = (0, height)

        diff, diff_img, diff_img_annotated = count_diff_SSIM(
            cropped_img1, cropped_img2, similarity_threshold)

        end = time.time()
        avg_time += (end - start)

    print("Time: %f" % (avg_time / float(num_iters)))

    print("min(%d) vs actual(%d) vs max(%d)" % (
        count_threshold[0], diff, count_threshold[1]))
