import os
import numpy as np
import cv2
import argparse
import yaml
import datetime

from classical_approach.process_image import count_diff_SSIM
import re


# format = "2019-11-04"
# date = format[8:10]

"""
Instructions:

If the difference image looks ok, press 'y'. else, press 'n'.
Pressing 'y' will store difference image in folder. 
Pressing 'n' will add to missed list, which you will have to find a better
negative to compare to. 
"""

positives_dir = "../../../axe_images/positives"
negatives_dir = "../../../axe_images/negative_compare"
diff_img_dir = "../../../axe_images/differences"
YAML_FILEPATH = "default_params.yaml"
file_format = "%Y-%m-%d_%H:%M:%S"

def get_date(file):
    date_match = re.search('\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}', file)
    date = datetime.datetime.strptime(date_match.group(), file_format)
    return date

def find_best_neg(negatives_map, negatives_list, pos_file):
    # '2019-10-31_23:59:59_diff:2563.png'
    target = get_date(pos_file)
    best_date = min(negatives_list, key=lambda date : abs(target-date))
    return negatives_map[best_date]

def main():
    # load tuned parameters
    with open(YAML_FILEPATH, 'r') as f:
        params = yaml.load(f)
    similarity_threshold = params["similarity_threshold"]
    count_threshold = (params["min_diff_pix"], params["max_diff_pix"])

    # store all negative filenames in mapping from date to the filename
    neg_date_to_file = dict()  # datetime -> datetime-specific filepath
    neg_dates = []  # all datetimes
    for neg_file in os.listdir(negatives_dir):
        if neg_file == ".DS_Store": continue
        date = get_date(neg_file)
        neg_dates.append(date)
        neg_date_to_file[date] = os.path.join(negatives_dir, neg_file)

    # loop through all positives, generate diff image, store
    wrong_diff = []
    all_pos = set(os.listdir(positives_dir))
    seen_pos = set(os.listdir(diff_img_dir))
    leftover = list(all_pos.difference(seen_pos))

    for pos_file in leftover:
        if pos_file == ".DS_Store": continue
        pos_path = os.path.join(positives_dir, pos_file) 
        pos = cv2.cvtColor(cv2.imread(pos_path), cv2.COLOR_BGR2RGB)

        neg_path = find_best_neg(neg_date_to_file, neg_dates, pos_file)
        print([pos_file, neg_path])

        neg = cv2.cvtColor(cv2.imread(neg_path), cv2.COLOR_BGR2RGB) 
        diff, diff_img, diff_img_annotated = count_diff_SSIM(pos, neg,
            similarity_threshold)

        cv2.imshow("pos, neg", np.hstack((pos, neg)))
        cv2.imshow("diff", diff_img_annotated)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('n'):
            wrong_diff.append(pos_path)
        elif key == ord('y'):
            new_path = os.path.join(diff_img_dir, pos_file)
            cv2.imwrite(new_path, diff_img)
        else:
            print("Wrong Key! Added to missed list")
            wrong_diff.append(pos_path)
    
    print("Files that were rejected:")
    print(wrong_diff)

if __name__ == "__main__":
    main()