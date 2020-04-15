import os
import sys
import numpy as np

ROOT_DIR = os.path.abspath("../../Mask_RCNN")
sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn import utils
import xml.etree.ElementTree

IMAGES_PATH = "/Users/Alvin/Documents/axe_images"
SOURCE = "axe_dataset"
CLASS = "axe_custom"
AXE_ID = 1  # 0 is for background!!!
CLASS_IDS = [0, AXE_ID]  # ['BG', 'axe']
TRAIN_DIR = os.path.join(IMAGES_PATH, "train")
VAL_DIR = os.path.join(IMAGES_PATH, "val")
TEST_DIR = os.path.join(IMAGES_PATH, "test")
LABELS_DIR = os.path.join(IMAGES_PATH, "labels")

class AxeDataset(utils.Dataset):
    def load_axe_images(self, path):
        self.add_class(source=SOURCE, class_id=AXE_ID, class_name=CLASS)

        images = os.listdir(path)
        for image_i in range(len(images)):
            imname = images[image_i]
            labelname = imname[:-3] + 'xml'
            impath = os.path.join(path, imname)
            labelpath = os.path.join(LABELS_DIR, labelname)
            self.add_image(source=SOURCE, image_id=image_i, path=impath, 
                labelpath=labelpath)


    def get_label(self, labelpath):
        et = xml.etree.ElementTree.parse(labelpath)
        root = et.getroot()
        
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        obj = root.find('object')
        bbox = obj.find('bndbox')
        xmin, xmax = int(bbox.find('xmin').text), int(bbox.find('xmax').text)
        ymin, ymax = int(bbox.find('ymin').text), int(bbox.find('ymax').text)

        return (xmin, ymin, xmax, ymax), (height, width)

    # def load_image(self, image_id):
    #     # no change from original load_image() function
    #     return super().load_image(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        bounds, imshape = self.get_label(info["labelpath"])
        xmin, ymin, xmax, ymax = bounds
        height, width = imshape

        # only one label to be applied
        mask = np.zeros(shape=(height, width, 1), dtype=np.uint8)
        mask[ymin:ymax, xmin:xmax, 0] = 1
        # np.ones([mask.shape[-1]], dtype=np.int32) * desired_id
        return mask.astype(np.bool), np.array([AXE_ID]).astype(np.int32)  # only detect axe