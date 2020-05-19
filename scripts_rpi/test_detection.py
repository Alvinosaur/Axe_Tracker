from maskrcnn_approach.AxeDetectModel import AxeDetectModel

import os
import cv2
import numpy as np
import matplotlib.pyplot  as plt

DEBUG = False

def generate_score(orig_img, detection, rings):
    detected_mask = detection["masks"][:,:,0].astype("uint8") * 255
    plt.imshow(detected_mask)
    plt.show()
    
    # generate bounding box with corners of detected mask
    _, cnt, hierarchy = cv2.findContours(detected_mask, 
        cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt = cnt[0]
    print(cnt)

    if DEBUG:
        print("Showing contours....")
        img = cv2.drawContours(orig_img, [cnt], -1, (0,255,0), 3)
        plt.imshow(img)
        plt.show()


    minx = cnt[cnt[:,:,0].argmin()][0]
    maxx = cnt[cnt[:,:,0].argmax()][0]
    miny = cnt[cnt[:,:,1].argmin()][0]
    maxy = cnt[cnt[:,:,1].argmax()][0]
    corners = [(minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)]
    scores = [5, 3, 1]
    assert(len(rings) == 3)
    # rings in order [inner: 5, middle: 3, outer: 1]
    for i in range(len(rings)):
        (cx, cy, r) = rings[i]
        # if axe tip within bounds of ring
        for (tx, ty) in corners:
            # if corner of bounding box lies within circle
            # return highest score
            if (tx - cx)**2 + (ty - cy)**2 <= r**2:
                print("Score: %d" %  scores[i])
                return scores[i]
    
    # if axe wasn't in any ring, return score 0
    print("No Axe Detected, Score: 0")
    return 0

test_dir = "/Users/Alvin/Documents/axe_images/test"
img_name = "2019-11-10_03:24:14_diff:1031_score:5.png"
original_image = cv2.imread(os.path.join(test_dir, img_name))
model = AxeDetectModel()
# model.test(original_image)
detection = model.generate_prediction(original_image)
generate_score(original_image, detection, rings=[])