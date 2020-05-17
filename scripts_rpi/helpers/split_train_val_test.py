import cv2
import os
import xml.etree.ElementTree
import shutil

import random

positives_dir = "../../../axe_images/new_positives"

train_dir = "../../../axe_images/train"
test_dir = "../../../axe_images/test"
val_dir = "../../../axe_images/val"

img_names = os.listdir(positives_dir)
img_names.remove('.DS_Store')
N = len(img_names)
random.shuffle(img_names)
# splits positives into 70% train, 15% val, 15% test
train_N = int(N * 0.7)
val_N = (N - train_N) // 2
train_imgs = img_names[:train_N]
val_imgs = img_names[train_N:train_N+val_N]
test_imgs = img_names[train_N+val_N:-1]

# copy over images
for img_name in train_imgs:
    old_img_path = os.path.join(positives_dir, img_name)
    new_img_path = os.path.join(train_dir, img_name)
    shutil.copyfile(old_img_path, new_img_path)

for img_name in val_imgs:
    old_img_path = os.path.join(positives_dir, img_name)
    new_img_path = os.path.join(val_dir, img_name)
    shutil.copyfile(old_img_path, new_img_path)

for img_name in test_imgs:
    old_img_path = os.path.join(positives_dir, img_name)
    new_img_path = os.path.join(test_dir, img_name)
    shutil.copyfile(old_img_path, new_img_path)