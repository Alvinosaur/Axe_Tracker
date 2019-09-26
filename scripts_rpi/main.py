#!/usr/bin/env python3

import argparse
import cv2
import numpy as  np
import os
import time

# from rpi_camera_interface import get_rpi_cam
from process_image import count_diff_SSIM

DEFAULT_PATH = '.'
DEFAULT_WIDTH, DEFAULT_HEIGHT = 1920, 1088
DEFAULT_SIMILARITY_THRESH = 10
DEFAULT_SLEEP = 1
DEFAULT_MIN_PIX = 300
DEFAULT_MAX_PIX = 400
verbose = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', dest='path', default=DEFAULT_PATH,
                        help='output path of detected axe images')
    parser.add_argument('--width', dest='width', default=DEFAULT_WIDTH, 
                        type=int, help='width of image')        
    parser.add_argument('--height', dest='height', default=DEFAULT_HEIGHT, 
                        type=int, help='height of image')      
    parser.add_argument('--sleep', dest='sleep', default=DEFAULT_SLEEP, 
                        type=int, help='sleep duration btwn taking new image')  
    parser.add_argument('--similarity_threshold', dest='similarity_threshold', 
                        default=DEFAULT_SIMILARITY_THRESH, type=int,
                        help='threshold for similarity measurement. The'+   
                        'higher the threshold, the more sensitive to' + 'differences.')
    parser.add_argument('--diff_min_pix', dest='diff_min_pix', 
                        default=DEFAULT_MIN_PIX, type=int,
                        help='MIN number of pixels diff btwn two images' + 
                        'to be considered axe')
    parser.add_argument('--diff_max_pix', dest='diff_max_pix', 
                        default=DEFAULT_MAX_PIX, type=int,
                        help='MAX number of pixels diff btwn two images' + 
                        'to be considered axe')
    args = parser.parse_args()
    output_folder = args.path
    width, height = args.width, args.height
    sleep_dur = args.sleep
    similarity_threshold = args.similarity_threshold
    count_threshold = (args.diff_min_pix, args.diff_max_pix)
    # cam = get_rpi_cam(width, height)
    cap = cv2.VideoCapture(0)
    prev, new =  None, np.empty((width, height, 3), dtype=np.uint8)

    while (True):
        # 2 second delay between grabbing images
        time.sleep(sleep_dur)

        # grab image
        # cam.capture(new, 'bgr')
        _, new  = cap.read()

        if prev is not None:
            diff = count_diff_SSIM(new, prev, width, height, 
                similarity_threshold)

            if (count_threshold[0] <= diff and
                diff <=  count_threshold[1]):

                # grab current time
                curr_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
                img_name = os.path.join(output_folder, curr_time + '.png')
                cv2.imwrite(img_name, new)
        
        prev = new
        
