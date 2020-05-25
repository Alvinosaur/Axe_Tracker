import argparse
import cv2
import numpy as  np
import os
import time
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from classical_approach.process_image import find_axe_tip
from helpers.ring_detect import fill_missing_rings, find_best_ring, draw_circles
import yaml


DEBUG = False
ONLY_USE_TEST = True
if ONLY_USE_TEST: print("Only evaluating performance on test dataset!")
YAML_FILEPATH = "params/default_params.yaml"
positives_dir = "../../../axe_images/positives"
test_dir = "../../../axe_images/test"
diff_img_dir = "../../../axe_images/differences"
# note: not ring_calib, which contains shrunken images
ring_calib_dir = "/Users/Alvin/Documents/axe_images/negative_compare"  
pos_to_neg_file = "pos_to_neg_mapping.npz"


def get_labeled_score(img_file):
    # 2019-11-14_00:45:06_diff:1133_score:1.png
    score = int(img_file[-5])
    return score

def generate_score(tip_pos, rings):
    tx, ty = tip_pos
    scores = [5, 3, 1]
    assert(len(rings) == 3)
    # rings in order [inner: 5, middle: 3, outer: 1]
    for i in range(len(rings)):
        (x, y, r) = rings[i]
        # if axe tip within bounds of ring
        if (tx - x)**2 + (ty - y)**2 <= r**2:
            return scores[i]
    
    # if axe wasn't in any ring, return score 0
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
    is_mini_img = False  # classical version runs on normal sized images
    if is_mini_img:
        ring_est_dist = params["ring_est_dist_mini"]
    else:
        ring_est_dist = params["ring_est_dist"]

    # Load positive-to-negative mapping
    data = np.load(pos_to_neg_file, allow_pickle=True)
    pos_to_neg = data["pos_to_neg"].item()

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
    diff_imgs = set(os.listdir(diff_img_dir))
    test_imgs = set(os.listdir(test_dir))
    
    if ONLY_USE_TEST: diff_imgs = diff_imgs.intersection(test_imgs)
    total_time = 0
    for img_file in diff_imgs:
        # both diff images and positives have same filename
        diff_img_path = os.path.join(diff_img_dir, img_file)
        orig_img_path = os.path.join(positives_dir, img_file)

        true_score = get_labeled_score(img_file)

        diff_img_color = cv2.imread(diff_img_path)
        diff_img = cv2.cvtColor(diff_img_color, cv2.COLOR_BGR2GRAY)
        orig_img = cv2.cvtColor(cv2.imread(orig_img_path), cv2.COLOR_BGR2RGB)

        # Find rings
        no_axe_img_path = os.path.join(ring_calib_dir, pos_to_neg[img_file])
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

        if ((inner_ring is None) and (middle_ring is None) and 
            (outer_ring is None)):
            missing_rings_imgs.append(img_file)
            continue
        
        # find rings and approximate true score
        rings = fill_missing_rings(inner_ring, middle_ring, outer_ring)

        start = time.time()
        axe_tip, box = find_axe_tip(diff_img)
        approx_score = generate_score(axe_tip, rings)
        end = time.time()
        total_time += (end - start)

        ## debug drawing
        if DEBUG:
            cv2.circle(diff_img_color, axe_tip, 10, (0, 0, 255), thickness=3)
            draw_circles(diff_img_color, rings)
            cv2.imshow('output', np.hstack([orig_img, diff_img_color]))
            cv2.waitKey(0)

    #     # update confusion matrix
        confusion_matrix[
            score_to_idx[true_score], 
            score_to_idx[approx_score]] += 1

    print("Average runtime: %.5f" % (total_time / float(len(diff_imgs))))
    print("Skipped images b/c missing rings:")
    print(missing_rings_imgs)

    print("confusion matrix:")
    print(confusion_matrix)

    # metrics for evaluating results wrt distribution of scores
    accuracy = np.sum(np.diag(confusion_matrix)) / len(diff_imgs)
    recall = np.diag(
        confusion_matrix / 
        np.reshape(np.sum(confusion_matrix,1), (4,1)))
    precision = np.diag(
        confusion_matrix / 
        np.reshape(np.sum(confusion_matrix,0), (4,1)))
    print(accuracy)
    print("Recall:")
    print(recall)
    print("Precision:")
    print(precision)

    print("normalized confusion matrix:")
    totals = np.reshape(np.sum(confusion_matrix, axis=1), (len(labels),1))
    norm_confusion_matrix = confusion_matrix / totals
    print(norm_confusion_matrix)

    plot_confusion_matrix(norm_confusion_matrix, labels,
        title="Normalized(by rows) Confusion Matrix: True v.s Approx Axe Scores")
    plot_confusion_matrix(confusion_matrix, labels,
        title="Confusion Matrix of Counts: True v.s Approx Axe Scores")
    # plot_confusion_matrix(norm_confusion_matrix, labels)