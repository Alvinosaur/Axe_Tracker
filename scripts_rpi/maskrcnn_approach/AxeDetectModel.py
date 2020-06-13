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

import maskrcnn_approach.AxeConfig as AxeConfig
import maskrcnn_approach.AxeDataset as AxeDataset

# Root directory of the project
ROOT_DIR = os.path.abspath("maskrcnn_approach/Mask_RCNN")
sys.path.append(ROOT_DIR)  # To find local version of the library
DEFAULT_WEIGHTS = "maskrcnn_approach/mask_rcnn_final_weights.h5"

# Import Mask RCNN
import mrcnn.model as modellib
from mrcnn import visualize

class InferenceConfig(AxeConfig.AxeConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class AxeDetectModel(object):
    def __init__(self, weights_path=DEFAULT_WEIGHTS, end_to_end=False):
        if end_to_end:
            self.class_ids_to_labels = {
                0: '-1',  # Background
                1: '0',
                2: '1',
                3: '3',
                4: '5'
            }
            self.class_labels_to_ids = {
                '0': 1,
                '1': 2,
                '3': 3,
                '5': 4
            }
        else: 
            AXE_ID = 1
            self.class_ids_to_labels = {
                0: '-1',
                AXE_ID: '1'
            }
            self.class_labels_to_ids = {
                '1': AXE_ID
            }
        self.inference_config = InferenceConfig(
            num_classes=len(self.class_ids_to_labels.keys()))
        self.dataset_train = AxeDataset.AxeDataset(
            class_ids_to_labels=self.class_ids_to_labels, 
            class_labels_to_ids=self.class_labels_to_ids,
            end_to_end=end_to_end)
        self.dataset_train.load_axe_images(AxeDataset.TRAIN_DIR)
        self.dataset_train.prepare()
        self.model_path = weights_path
    
        # Recreate the model in inference mode
        self.model = modellib.MaskRCNN(mode="inference", 
                                config=self.inference_config,
                                model_dir=ROOT_DIR + "/logs")
        self.model.load_weights(self.model_path, by_name=True)

    def generate_prediction(self, original_image, return_all=False, 
            show_results=False):
        results = self.model.detect([original_image], verbose=1)
        r = results[0]
        if return_all:
            if show_results:
                visualize.display_instances(original_image, r['rois'], 
                    r['masks'], r['class_ids'], 
                    AxeDetectModel.dataset_train.class_names, 
                    r['scores'], figsize=(8, 8))
            
            return r
        else:
            try:
                best_i = np.argmax(r['scores'])
                best_roi = np.reshape(r['rois'][best_i], newshape=(1,4))
                h, w, _ = r['masks'].shape
                best_mask = np.reshape(r['masks'][:,:,best_i], newshape=(h,w,1))
                best_class_id = np.array([r['class_ids'][best_i]])
                best_score = np.array([r['scores'][best_i]])

                if show_results:
                    visualize.display_instances(original_image, best_roi, 
                        best_mask, best_class_id, 
                        AxeDetectModel.dataset_train.class_names, 
                        best_score, figsize=(8, 8))

                best_res = {
                    'rois': best_roi,
                    'masks': best_mask,
                    'class_ids': best_class_id,
                    'scores': best_score
                }
                return best_res
            except: 
                print("No axe detected!")
                return None

    def test(self, img):
        res = self.generate_prediction(img)
        visualize.display_instances(img, res['rois'], res['masks'], 
                    res['class_ids'], self.dataset_train.class_names, 
                    res['scores'], figsize=(8, 8))
        

def test(model):
    full_size_convoluted_path = "/Users/Alvin/Documents/axe_images/convoluted"
    convoluted_path = "/Users/Alvin/Documents/axe_images/resized_convoluted"
    test_dir = "/Users/Alvin/Documents/axe_images/test"
    convoluted_ims = os.listdir(convoluted_path)

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

    for i in range(10):
        # Load some test image
        # image_id = random.choice(dataset_test.image_ids)
        # original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        #     modellib.load_image_gt(dataset_val, inference_config, 
        #                         image_id, use_mini_mask=False)
        # visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
        #                             dataset_train.class_names, figsize=(8, 8))

        # img_name = random.choice(convoluted_ims)
        img_name = "2019-11-10_20:10:43_diff:1267_score:1.png"
        original_image = cv2.imread(os.path.join(test_dir, img_name))
        cv2.imshow('original', cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        cv2.waitKey(0)

        # run image through network to predict axe location
        results = model.detect([original_image], verbose=1)
        r = results[0]
        show_all_boxes = False
        if show_all_boxes:
            visualize.display_instances(original_image, r['rois'], r['masks'], 
                r['class_ids'], dataset_val.class_names, r['scores'], figsize=(8, 8))
        else:
            try:
                best_i = np.argmax(r['scores'])
                best_roi = np.reshape(r['rois'][best_i], newshape=(1,4))
                h, w, _ = r['masks'].shape
                best_mask = np.reshape(r['masks'][:,:,best_i], newshape=(h,w,1))
                best_class_id = np.array([r['class_ids'][best_i]])
                best_score = np.array([r['scores'][best_i]])
                visualize.display_instances(original_image, best_roi, best_mask, 
                    best_class_id, dataset_val.class_names, best_score, 
                    figsize=(8, 8))
            except: 
                print("No axe detected!")
                continue

# if __name__ == "__main__":
#     test()