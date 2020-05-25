import os
import numpy as np
import cv2
import argparse
import yaml
import datetime

from classical_approach.find_diff_images import get_date, find_best_neg
from helpers.resize_relabel_images import manually_resize
import re

positives_dir = "../../../axe_images/positives"
negatives_dir = "../../../axe_images/negative_compare"
mini_negatives_dir = "../../../axe_images/ring_calib"


def main():
    # store all negative filenames in mapping from date to the filename
    neg_date_to_file = dict()  # datetime -> datetime-specific filepath
    neg_dates = []  # all datetimes
    for neg_file in os.listdir(negatives_dir):
        if neg_file == ".DS_Store": continue
        date = get_date(neg_file)
        neg_dates.append(date)
        neg_date_to_file[date] = neg_file

    # loop through all positives, generate diff image, store
    all_pos = os.listdir(positives_dir)
    pos_to_neg = dict()

    for pos_file in all_pos:
        if pos_file == ".DS_Store": continue
        pos_path = os.path.join(positives_dir, pos_file)
        neg_file = find_best_neg(neg_date_to_file, neg_dates, pos_file)
        mini_neg_img = manually_resize(os.path.join(negatives_dir, neg_file))
        cv2.imwrite(os.path.join(mini_negatives_dir, neg_file), mini_neg_img)
        pos_to_neg[pos_file] = neg_file

    np.savez("pos_to_neg_mapping", pos_to_neg=pos_to_neg)


if __name__ == "__main__":
    main()