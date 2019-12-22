#!/usr/bin/env python3

import argparse
import cv2
import numpy as  np
import os
import time

# from rpi_camera_interface import get_rpi_cam
import picamera
from process_image import count_diff_SSIM
import yaml

verbose = True
YAML_FILEPATH = "default_params.yaml"
CAM_FRAMERATE =  24
file_format = "%Y-%m-%d_%H:%M:%S"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', dest='path', default=YAML_FILEPATH,
                        help='path to yaml file with params')
    args = parser.parse_args()
    with open(args.path, 'r') as f:
        params = yaml.load(f)

    if verbose: print(params)    
    output_folder = params["image_folder_path"]
    width, height = params["image_width"], params["image_height"]
    sleep_dur = params["sleep_dur"]
    similarity_threshold = params["similarity_threshold"]
    count_threshold = (params["min_diff_pix"], params["max_diff_pix"])
    crop_bounds_w = (params["image_min_w"], params["image_max_w"])
    crop_bounds_h = (params["image_min_h"], params["image_max_h"])
    min_cap_rate = params["min_cap_rate"]
    
    # cam = get_rpi_cam(width, height)
    camera = picamera.PiCamera()
    camera.resolution = (width, height)
    camera.framerate = CAM_FRAMERATE
    time.sleep(2) # sleep for 2 seconds to initialize camera hardware
    cur_time = time.time()

    # cap = cv2.VideoCapture(0)
    prev_bgr, new_bgr =  None, np.empty((height, width, 3), dtype=np.uint8)
    saved_prev = False

    while (True):
        cur_time = time.time()
        # 2 second delay between grabbing images
        time.sleep(sleep_dur)

        # grab image
        camera.capture(new_bgr, 'bgr')
        # _, new  = cap.read()

        # if saved previous image, don't try to compare to axeless image
        if (prev_bgr is not None) and (not saved_prev):
            diff = count_diff_SSIM(new_bgr, prev_bgr, width, height, 
                crop_bounds_w, crop_bounds_h, similarity_threshold)
            if verbose: print("new image pair diff: %d" % diff)

            if count_threshold[0] <= diff:
                # grab current time
                curr_time = time.strftime(file_format, time.gmtime())
                img_name = os.path.join(output_folder, curr_time)
                new_img_name = img_name + '_new' + '.png'
                old_img_name = img_name + '_prev' + '.png'
                new_rgb = cv2.cvtColor(new_bgr, cv2.COLOR_BGR2RGB)[
                    crop_bounds_h[0]:crop_bounds_h[1], 
                    crop_bounds_w[0]:crop_bounds_w[1]]
                prev_rgb = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2RGB)[
                    crop_bounds_h[0]:crop_bounds_h[1], 
                    crop_bounds_w[0]:crop_bounds_w[1]]
 
                cv2.imwrite(new_img_name, new_rgb)
                cv2.imwrite(old_img_name, prev_rgb)

                saved_prev = True
        else:
            saved_prev = False

        prev_bgr = np.copy(new_bgr)
        
