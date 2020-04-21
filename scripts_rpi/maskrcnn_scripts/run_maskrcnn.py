import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import xml.etree.ElementTree

import AxeConfig
import AxeDataset

# Root directory of the project
ROOT_DIR = os.path.abspath("../../Mask_RCNN")
sys.path.append(ROOT_DIR)  # To find local version of the library

# Import Mask RCNN
import mrcnn.model as modellib
from mrcnn import visualize

WEIGHTS_PATH = "mask_rcnn_final_weights.h5"

# Training dataset
dataset_train = AxeDataset.AxeDataset()
dataset_train.load_axe_images(AxeDataset.TRAIN_DIR)
dataset_train.prepare()

# Validation dataset
dataset_val = AxeDataset.AxeDataset()
dataset_val.load_axe_images(AxeDataset.VAL_DIR)
dataset_val.prepare()

# Validation dataset
dataset_test = AxeDataset.AxeDataset()
dataset_test.load_axe_images(AxeDataset.TEST_DIR)
dataset_test.prepare()

class InferenceConfig(AxeConfig.AxeConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig(2)

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=ROOT_DIR + "/logs")
model_path = WEIGHTS_PATH
model.load_weights(model_path, by_name=True)

for i in range(10):
    # Load some test image
    image_id = random.choice(dataset_test.image_ids)
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_val, inference_config, 
                            image_id, use_mini_mask=False)
    visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
                                dataset_train.class_names, figsize=(8, 8))

    # run image through network to predict axe location
    results = model.detect([original_image], verbose=1)
    print("Displaying predicted:")
    r = results[0]
    show_all_boxes = False
    if show_all_boxes:
        visualize.display_instances(original_image, r['rois'], r['masks'], 
            r['class_ids'], dataset_val.class_names, r['scores'], figsize=(8, 8))
    else:
        best_i = np.argmax(r['scores'])
        best_roi = np.reshape(r['rois'][best_i], newshape=(1,4))
        h, w, _ = r['masks'].shape
        best_mask = np.reshape(r['masks'][:,:,best_i], newshape=(h,w,1))
        best_class_id = np.array([r['class_ids'][best_i]])
        best_score = np.array([r['scores'][best_i]])
        visualize.display_instances(original_image, best_roi, best_mask, 
            best_class_id, dataset_val.class_names, best_score, figsize=(8, 8))
