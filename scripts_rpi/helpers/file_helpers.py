import csv
import cv2
import numpy as np
import os

img_folder = ""

def read_csv_file(path):
    with open(path, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        data = [row for row in csv_reader]
    return data

def get_img_names(folder_path):
    global img_folder
    img_folder = folder_path
    return os.listdir(folder_path)

def load_image(img_name):
    img_path = os.path.join(img_folder, img_name)
    try:
        img = cv2.imread(img_path)
        img = cv2.pyrDown(img, dstsize=(img.shape[1] // 2, img.shape[0] // 2))
        num_features = img.shape[0] * img.shape[1] * 3
        return img.reshape((num_features, 1))

    except Exception as e:
        print("Error loading image because %s" % e)
        return None
    

def parse_true_label(img_name):
    # 2019-11-13_01:00:08_diff:1128_score:0.png
    try:
        return int(img_name[-5])
    except:
        print("Image not labeled: %s" % img_name)
        return None
