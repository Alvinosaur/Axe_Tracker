import argparse
import cv2
import numpy as  np
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
if ONLY_USE_TEST: print("Only evaluating performance on test dataset!")
YAML_FILEPATH = "params/default_params.yaml"
full_size_dir = "/Users/Alvin/Documents/axe_images/positives"
positives_dir = "/Users/Alvin/Documents/axe_images/new_positives"
test_dir = "/Users/Alvin/Documents/axe_images/test"

# /Users/Alvin/Documents/axe_images/test/2019-11-09_03:39:11_diff:1129_score:3.png


def get_labeled_score(img_file):
    # 2019-11-14_00:45:06_diff:1133_score:1.png
    score = int(img_file[-5])
    return score

def generate_score(orig_img, detection, rings):
    detected_mask = detection["masks"][:,:,0].astype("uint8") * 255
    # if DEBUG:
    #     plt.imshow(detected_mask)
    #     plt.show()
    
    # generate bounding box with corners of detected mask
    _, cnt, hierarchy = cv2.findContours(detected_mask, 
        cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    cnt = cnt[0].reshape(-1,2)
    minxi, maxxi = cnt[:,0].argmin(), cnt[:,0].argmax()
    minx, maxx = cnt[minxi, 0], cnt[maxxi, 0]
    minyi, maxyi = cnt[:,1].argmin(), cnt[:,1].argmax()
    miny, maxy = cnt[minyi, 1], cnt[maxyi, 1]
    corners = [(minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)]
    
    true_score = None
    scores = [5, 3, 1]
    assert(len(rings) == 3)
    # rings in order [inner: 5, middle: 3, outer: 1]
    for i in range(len(rings)):
        (cx, cy, r) = rings[i]
        # if axe tip within bounds of ring
        if true_score is not None: break
        for (tx, ty) in corners:
            # if corner of bounding box lies within circle
            # return highest score
            if (tx - cx)**2 + (ty - cy)**2 <= r**2:
                print("Score: %d" %  scores[i])
                true_score = scores[i]

    if DEBUG:
        print("Showing contour corners...")
        image = np.copy(orig_img)
        for (x, y) in corners:
            image = cv2.circle(image, (x,y), radius=1, color=(0, 0, 255), 
                thickness=-1)
        draw_circles(image, rings, thickness=1)
        plt.imshow(image)
        plt.show()

    # scores = [5, 3, 1]
    # assert(len(rings) == 3)
    # # rings in order [inner: 5, middle: 3, outer: 1]
    # for i in range(len(rings)):
    #     (cx, cy, r) = rings[i]
    #     # if axe tip within bounds of ring
    #     for (tx, ty) in corners:
    #         # if corner of bounding box lies within circle
    #         # return highest score
    #         if (tx - cx)**2 + (ty - cy)**2 <= r**2:
    #             print("Score: %d" %  scores[i])
    #             true_score = scores[i]
    #             break
                # return scores[i]
    
    # if axe wasn't in any ring, return score 0
    print("No Axe Detected, Score: 0")
    return 0

def plot_confusion_matrix(mat, labels, title):
    df_cm = pd.DataFrame(mat, labels, labels)
    # plt.figure(figsize=(10,7))
    sn.set(font_scale=1.4) # for label size
    cmap = sn.cm.rocket_r
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cmap=cmap) # font size
    plt.xlabel("Approx Scores")
    plt.ylabel("True Scores")
    plt.title(title)
    plt.show()


def plot_score_distib(diff_img_files):
    scores = [0, 1, 3, 5]
    counts = np.zeros(len(scores))
    score_to_idx = {
        0: 0,
        1: 1,
        3: 2,
        5: 3
    }
    for img_file in diff_img_files:
        true_score = get_labeled_score(img_file)
        counts[score_to_idx[true_score]] += 1

    total = len(diff_img_files)
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
    ring_est_dist = params["ring_est_dist"]

    # MaskRCNN detection model
    # mrcnn_model = AxeDetectModel(weights_path="maskrcnn_approach/mask_rcnn_axe_0060.h5")
    # mrcnn_model = AxeDetectModel(weights_path="maskrcnn_approach/mask_rcnn_axe_0060.h5")
    mrcnn_model = AxeDetectModel()

    # confusion matrix to understand results
    labels = [0, 1, 3, 5]
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

    if ONLY_USE_TEST: pos_imgs = pos_imgs.intersection(test_imgs)
    if ".DS_Store" in pos_imgs: pos_imgs.remove(".DS_Store")

    no_detect_count = 0
    total_time = 0
    for img_file in pos_imgs:
        orig_img_path = os.path.join(positives_dir, img_file)

        true_score = get_labeled_score(img_file)
        print("True Score: %d" % true_score)
        orig_img = cv2.imread(orig_img_path)
        # mrcnn_model.test(orig_img)
        full_size_img = cv2.cvtColor(
            cv2.imread(os.path.join(full_size_dir, img_file)),
            cv2.COLOR_BGR2RGB)

        # Find rings
        inner_ring = None  # inner ring is too unreliable with shrunken image
        middle_ring = find_best_ring(full_size_img, 
            R_bounds2, G_bounds2, B_bounds2,
            hough_p1, hough_p2, hough_dp)
        outer_ring = find_best_ring(full_size_img, 
            R_bounds3, G_bounds3, B_bounds3,
            hough_p1, hough_p2, hough_dp)

        if ((inner_ring is None) and (middle_ring is None) and 
            (outer_ring is None)):
            missing_rings_imgs.append(img_file)
            continue
        
        # find rings and approximate true score
        rings = fill_missing_rings(inner_ring, middle_ring, outer_ring, 
            out_to_mid=ring_est_dist, mid_to_in=ring_est_dist)

        for ri, (cx,cy,r) in enumerate(rings):
            rings[ri] = transform_ring(rings[ri])
        
        start = time.time()
        detection = mrcnn_model.generate_prediction(orig_img)
        try:
            approx_score = generate_score(orig_img, detection, rings)
        except Exception as e:
            # no axe detected
            print("NO AXE DETECTED for img %s because %s" % (img_file, e))
            approx_score = 0
            no_detect_count += 1
            if DEBUG:
                plt.imshow(orig_img)
                plt.show()
        end = time.time()
        total_time += (end - start)

        # debug drawing
        # if DEBUG:
        #     try:
        #         draw_circles(orig_img, rings, thickness=1)
        #     except Exception as e:
        #         print(img_file)
        #         print("RING ISSUE: %s" % e)

        #     cv2.imshow("Detected Rings", orig_img)
        #     cv2.waitKey(0)

        # update confusion matrix
        confusion_matrix[
            score_to_idx[true_score], 
            score_to_idx[approx_score]] += 1

    print("Average runtime: %.2f" % (total_time / float(len(pos_imgs))))

    print("Skipped images b/c missing rings:")
    print(missing_rings_imgs)

    no_detect_rate = no_detect_count / float(len(pos_imgs))
    print("Percent No Detections: %.2f" % no_detect_rate)

    print("confusion matrix:")
    print(confusion_matrix)

    # metrics for evaluating results wrt distribution of scores
    accuracy = np.sum(np.diag(confusion_matrix)) / len(pos_imgs)
    recall = np.diag(
        confusion_matrix / 
        np.reshape(np.sum(confusion_matrix,1), (4,1)))
    precision = np.diag(
        confusion_matrix / np.sum(confusion_matrix))
    print("Accuracy:")
    print(accuracy)
    print("Recall:")
    print(recall)
    print("Precision")
    print(precision)

    print("normalized confusion matrix:")
    totals = np.reshape(np.sum(confusion_matrix, axis=1), (len(labels),1))
    norm_confusion_matrix = confusion_matrix / totals
    print(norm_confusion_matrix)

    plot_confusion_matrix(norm_confusion_matrix, labels,
        title="Normalized(by rows) Confusion Matrix: True v.s Approx Axe Scores")
    # plot_confusion_matrix(norm_confusion_matrix, labels)