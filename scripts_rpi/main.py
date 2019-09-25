#!/usr/bin/env python3

import argparse
import cv2
import numpy as  np
import os
import time

from rpi_camera_interface import get_rpi_cam
from process_image import get_diff_SSIM

DEFAULT_PATH = '/home/pi/AxeImgs'
DEFAULT_WIDTH, DEFAULT_HEIGHT = 1920, 1088
DEFAULT_SIMILARITY_THRESH = (0.85, 0.95)
DEFAULT_SLEEP = 2
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
    args = parser.parse_args()
    output_folder = args.path
    width, height = args.width, args.height
    sleep_dur = args.sleep
    cam = get_rpi_cam(width, height)
    prev, new =  None, np.empty((width, height, 3), dtype=np.uint8)

    while (True):
        # 2 second delay between grabbing images
        time.sleep(sleep_dur)

        # grab image
        cam.capture(new, 'bgr')

        if prev is not None:
            diff = get_diff_SSIM(new, prev, width, height)
            if verbose:
                print("Difference: %f" % diff)
                cv2.imshow("New Image", new)
                cv2.imshow("Prev Image", prev)

            if (DEFAULT_SIMILARITY_THRESH[0] <= diff and
                diff <=  DEFAULT_SIMILARITY_THRESH[1]):

                # grab current time
                curr_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
                img_name = os.path.join(output_folder, curr_time + '.png')
                cv2.imwrite(img_name, new)
        
        prev = new
        
