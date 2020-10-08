import argparse
import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt

from maskrcnn_approach.AxeDetectModel import AxeDetectModel, test as test_mcrnn
from maskrcnn_approach.AxeDataset import AxeDataset
from helpers.ring_detect import (
    fill_missing_rings, find_best_ring, draw_circles, NoRingError)
from helpers.resize_relabel_images import get_warp, transform_ring
import yaml

DEBUG = False
END_TO_END = False
YAML_FILEPATH = "params/default_params.yaml"
positives_dir = "../axe_images/new_positives"
test_dir = "../axe_images/test"
ring_calib_dir = "../axe_images/ring_calib"
pos_to_neg_file = "pos_to_neg_mapping.npz"

# order matters, start from innermost ring(highest score) and work outwards
nonzero_scores = [5, 3, 1]
labels = [0, 1, 3, 5]


def get_labeled_score(img_file):
    """Parse labeled image's filename for score. Example:
    2019-11-14_00:45:06_diff:1133_score:1.png

    Args:
        img_file (str): image filename

    Returns:
        int: score for this image
    """
    score = int(img_file[-5])
    return score


def generate_score(detection, rings):
    """Extract bounds of MaskRCNN's detected bounding box and find
    the innermost ring that any corner lies in.

    Args:
        detection (dict): output of MASKRCNNN, filtered so only most confident
        detection remains.
        rings (list): [(cx,cy,r)...] describing the three rings of bullseye

    Returns:
        (int, list): score and corners(x,y) of detected axe as [top-left,
        top-right, bot-left, bot-right]
    """
    # read in mask representing detected axe tip and bring to integer pixel values
    detected_mask = detection["masks"][:, :, 0].astype("uint8") * 255

    # extract corners of mask
    _, cnt, hierarchy = cv2.findContours(detected_mask,
                                         cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = cnt[0].reshape(-1, 2)
    minxi, maxxi = cnt[:, 0].argmin(), cnt[:, 0].argmax()
    minx, maxx = cnt[minxi, 0], cnt[maxxi, 0]
    minyi, maxyi = cnt[:, 1].argmin(), cnt[:, 1].argmax()
    miny, maxy = cnt[minyi, 1], cnt[maxyi, 1]
    corners = [(minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)]

    assert(len(rings) == 3)
    # rings in order [inner: 5, middle: 3, outer: 1]
    for i in range(len(rings)):
        (cx, cy, r) = rings[i]
        # if any corner of bounding box lies within ring
        for (tx, ty) in corners:
            if (tx - cx)**2 + (ty - cy)**2 <= r**2:
                if DEBUG:
                    print("Score: %d" % nonzero_scores[i])
                return nonzero_scores[i], corners

    # if axe wasn't in any ring, return score 0
    if DEBUG:
        print("No Axe Detected, Score: 0")
    return 0, corners


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', dest='path', default=YAML_FILEPATH,
                        help='path to yaml file with params')
    args = parser.parse_args()
    with open(args.path, 'r') as f:
        params = yaml.load(f)

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

    # Load positive-to-negative mapping
    data = np.load(pos_to_neg_file, allow_pickle=True)
    pos_to_neg = data["pos_to_neg"].item()

    # MaskRCNN detection model
    mrcnn_model = AxeDetectModel(
        weights_path="maskrcnn_approach/mask_rcnn_tip_detection.h5", end_to_end=END_TO_END)

    # confusion matrix to understand results
    score_to_idx = {
        0: 0,
        1: 1,
        3: 2,
        5: 3
    }
    # rows are true label, columns are approx
    confusion_matrix = np.zeros((len(labels), len(labels)))

    missing_rings_imgs = []

    # already have difference images stored in differences folder
    pos_imgs = os.listdir(positives_dir)
    if ".DS_Store" in pos_imgs:
        pos_imgs.remove(".DS_Store")

    total_time = 0
    # pos_imgs = pos_imgs[0:2]
    for img_file in pos_imgs:
        if DEBUG:
            print("Img: %s" % img_file)
        orig_img_path = os.path.join(positives_dir, img_file)

        orig_img = cv2.imread(orig_img_path)
        orig_img_RGB = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        # mrcnn_model.test(orig_img)

        # NOTE: in real system, we do not use the actual image with axe for ring
        # detection since axe interferes with color-thresholding
        # rings are predetermined every 30 minutes and stored in a file
        start = time.time()
        inner_ring = find_best_ring(orig_img_RGB,
                                    R_bounds1, G_bounds1, B_bounds1,
                                    hough_p1, hough_p2, hough_dp, ring_est_dist)
        middle_ring = find_best_ring(orig_img_RGB,
                                     R_bounds2, G_bounds2, B_bounds2,
                                     hough_p1, hough_p2, hough_dp, 2 * ring_est_dist)
        outer_ring = find_best_ring(orig_img_RGB,
                                    R_bounds3, G_bounds3, B_bounds3,
                                    hough_p1, hough_p2, hough_dp, 3 * ring_est_dist)

        # find rings and approximate true score
        rings = fill_missing_rings(inner_ring, middle_ring, outer_ring,
                                   out_to_mid=ring_est_dist, mid_to_in=ring_est_dist)

        detection = mrcnn_model.generate_prediction(orig_img)
        true_score = AxeDataset.score_from_labelpath(img_file)

        end = time.time()
        total_time += (end - start)

        try:
            approx_score, corners = generate_score(
                detection, rings)
            no_detect = False
        except Exception as e:
            # no axe detected
            print("NO AXE DETECTED for img %s because %s" % (img_file, e))
            approx_score = 0
            no_detect = True
            corners = []

        print("True: %s, Approx: %d" % (true_score, approx_score))

    print("Average runtime: %.2f" % (total_time / float(len(pos_imgs))))
