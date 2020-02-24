import os
import numpy as np
import cv2
import argparse
import yaml

from process_image import count_diff_SSIM


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

with open(YAML_FILEPATH, 'r') as f:
    params = yaml.load(f)

similarity_threshold = params["similarity_threshold"]
count_threshold = (params["min_diff_pix"], params["max_diff_pix"])
# store all negative filenames in mapping from date to the filename
all_negatives = dict()
for neg_file in os.listdir(negatives_dir):
    if neg_file == ".DS_Store": continue
    date = neg_file[8:10]
    all_negatives[date] = os.path.join(negatives_dir, neg_file)

# loop through all positives, generate diff image, store
wrong_diff = []
for pos_file in os.listdir(positives_dir):
    if pos_file == ".DS_Store": continue
    pos_path = os.path.join(positives_dir, pos_file) 
    pos = cv2.cvtColor(cv2.imread(pos_path), cv2.COLOR_BGR2RGB)

    date = pos_file[8:10]
    neg_file = all_negatives[date]
    neg = cv2.cvtColor(cv2.imread(neg_file), cv2.COLOR_BGR2RGB)

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
