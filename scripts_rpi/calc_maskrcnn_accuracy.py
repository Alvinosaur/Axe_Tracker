import argparse
import cv2
import numpy as np
import os
import time
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from maskrcnn_approach.AxeDetectModel import AxeDetectModel, test as test_mcrnn
from helpers.ring_detect import (
    fill_missing_rings, find_best_ring, draw_circles, NoRingError)
from helpers.resize_relabel_images import get_warp, transform_ring
import yaml


DEBUG = True
ONLY_USE_TEST = True
END_TO_END = False
if ONLY_USE_TEST:
    print("Only evaluating performance on test dataset!")
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


def plot_confusion_matrix(mat, labels, title):
    df_cm = pd.DataFrame(mat, labels, labels)
    # plt.figure(figsize=(10,7))
    sn.set(font_scale=1.4)  # for label size
    cmap = sn.cm.rocket_r
    sn.heatmap(df_cm, annot=True, annot_kws={
               "size": 16}, cmap=cmap)  # font size
    plt.xlabel("Approx Scores")
    plt.ylabel("True Scores")
    plt.title(title)
    plt.show()


def plot_score_distib(labeled_imgnames):
    scores = [0, 1, 3, 5]
    counts = np.zeros(len(scores))
    score_to_idx = {
        0: 0,
        1: 1,
        3: 2,
        5: 3
    }
    for img_file in labeled_imgnames:
        true_score = get_labeled_score(img_file)
        counts[score_to_idx[true_score]] += 1

    total = len(labeled_imgnames)
    counts = counts / total

    x_ind = range(len(scores))
    plt.xticks(x_ind, scores)
    plt.yticks(counts)
    plt.bar(x_ind, counts)
    plt.xlabel('Scores')
    plt.ylabel('Proportion of Total Dataset')
    plt.title('Score Distribution in Dataset')
    plt.show()


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
    # mrcnn_model = AxeDetectModel(weights_path="maskrcnn_approach/mask_rcnn_axe_0060.h5")
    if END_TO_END:
        mrcnn_model = AxeDetectModel(
            weights_path="maskrcnn_approach/mask_rcnn_end_to_end.h5", end_to_end=END_TO_END)
    else:
        mrcnn_model = AxeDetectModel(
            weights_path="maskrcnn_approach/mask_rcnn_tip_detection.h5", end_to_end=END_TO_END)
    # mrcnn_model = AxeDetectModel()

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
    pos_imgs = set(os.listdir(positives_dir))
    test_imgs = set(os.listdir(test_dir))

    if ONLY_USE_TEST:
        pos_imgs = pos_imgs.intersection(test_imgs)
    if ".DS_Store" in pos_imgs:
        pos_imgs.remove(".DS_Store")

    no_detect_count = 0
    total_time = 0
    for img_file in pos_imgs:
        print("Img: %s" % img_file)
        orig_img_path = os.path.join(positives_dir, img_file)

        true_score = get_labeled_score(img_file)
        orig_img = cv2.imread(orig_img_path)
        orig_img_RGV = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        # mrcnn_model.test(orig_img)

        if END_TO_END:
            detection = mrcnn_model.generate_prediction(orig_img,
                                                        show_results=DEBUG)
            try:
                class_id = int(detection['class_ids'][0])
                approx_score = int(mrcnn_model.class_ids_to_labels[class_id])
                # safety check in case model predicts background for some reason
                assert(approx_score != -1)
                no_detect = False
            except Exception as e:
                # no axe detected
                print("NO AXE DETECTED for img %s because %s" % (img_file, e))
                approx_score = 0
                no_detect_count += 1
                no_detect = True
        else:
            # Find rings
            # run ring detection only once on image without any axe
            no_axe_img_path = os.path.join(
                ring_calib_dir, pos_to_neg[img_file])
            no_axe_img = cv2.cvtColor(cv2.imread(no_axe_img_path),
                                      cv2.COLOR_BGR2RGB)
            inner_ring = find_best_ring(no_axe_img,
                                        R_bounds1, G_bounds1, B_bounds1,
                                        hough_p1, hough_p2, hough_dp, ring_est_dist)
            middle_ring = find_best_ring(no_axe_img,
                                         R_bounds2, G_bounds2, B_bounds2,
                                         hough_p1, hough_p2, hough_dp, 2*ring_est_dist)
            outer_ring = find_best_ring(no_axe_img,
                                        R_bounds3, G_bounds3, B_bounds3,
                                        hough_p1, hough_p2, hough_dp, 3*ring_est_dist)

            # find rings and approximate true score
            rings = fill_missing_rings(inner_ring, middle_ring, outer_ring,
                                       out_to_mid=ring_est_dist, mid_to_in=ring_est_dist)

            # for ri, (cx,cy,r) in enumerate(rings):
            #     rings[ri] = transform_ring(rings[ri])

            detection = mrcnn_model.generate_prediction(orig_img)
            try:
                approx_score, corners = generate_score(
                    detection, rings)
                no_detect = False
            except Exception as e:
                # no axe detected
                print("NO AXE DETECTED for img %s because %s" % (img_file, e))
                approx_score = 0
                no_detect_count += 1
                no_detect = True
                corners = []

            # debug drawing
            if DEBUG:
                try:
                    draw_circles(orig_img_RGV, rings, thickness=1)
                except Exception as e:
                    print(img_file)
                    print("RING ISSUE: %s" % e)

                for (x, y) in corners:
                    orig_img_RGV = cv2.circle(orig_img_RGV, (x, y), radius=1,
                                              color=(0, 0, 255), thickness=-1)

                if true_score != approx_score and not no_detect:
                    print("Neg Img: %s" % pos_to_neg[img_file])
                    print("True, Approx: (%d, %d)" %
                          (true_score, approx_score))
                    plt.imshow(orig_img_RGV)
                    plt.show()

        # update confusion matrix
        if not no_detect:
            confusion_matrix[
                score_to_idx[true_score],
                score_to_idx[approx_score]] += 1

    # print("Average runtime: %.2f" % (total_time / float(len(pos_imgs))))

    print("Skipped images b/c missing rings:")
    print(missing_rings_imgs)

    no_detect_rate = no_detect_count / float(len(pos_imgs))
    print("Percent No Detections: %.2f" % no_detect_rate)

    print("confusion matrix:")
    print(confusion_matrix)

    # metrics for evaluating results wrt distribution of scores
    accuracy = np.sum(np.diag(confusion_matrix)) / len(pos_imgs)
    recall = np.diag(confusion_matrix /
                     np.reshape(np.sum(confusion_matrix, 1), (4, 1)))
    precision = np.diag(confusion_matrix /
                        np.reshape(np.sum(confusion_matrix, 0), (4, 1)))
    print("Accuracy:")
    print(accuracy)
    print("Recall:")
    print(recall)
    print("Precision")
    print(precision)

    print("normalized confusion matrix:")
    totals = np.reshape(np.sum(confusion_matrix, axis=1), (len(labels), 1))
    norm_confusion_matrix = confusion_matrix / totals
    print(norm_confusion_matrix)

    plot_confusion_matrix(norm_confusion_matrix, labels,
                          title="Normalized(by rows) Confusion Matrix: True v.s Approx Axe Scores")
    plot_confusion_matrix(confusion_matrix, labels,
                          title="Confusion Matrix of Counts: True v.s Approx Axe Scores")
    # plot_confusion_matrix(norm_confusion_matrix, labels)
