import xml.etree.ElementTree
from mrcnn import utils
import os
import sys
import numpy as np

ROOT_DIR = os.path.abspath("Mask_RCNN")
sys.path.append(ROOT_DIR)  # To find local version of the library


IMAGES_PATH = "../axe_images"
SOURCE = "axe_dataset"
AXE_ID = 1  # 0 is for background!!!
TRAIN_DIR = os.path.join(IMAGES_PATH, "train")
VAL_DIR = os.path.join(IMAGES_PATH, "val")
TEST_DIR = os.path.join(IMAGES_PATH, "test")
LABELS_DIR = os.path.join(IMAGES_PATH, "labels")


class AxeDataset(utils.Dataset):
    def __init__(self, class_ids_to_labels, class_labels_to_ids, end_to_end):
        self.end_to_end = end_to_end
        self.class_ids_to_labels = class_ids_to_labels
        self.class_labels_to_ids = class_labels_to_ids
        super().__init__(class_map=None)

    def load_axe_images(self, path):
        for id, name in self.class_ids_to_labels.items():
            if id == 0:
                continue  # don't include background, it's there by default
            self.add_class(source=SOURCE, class_id=id, class_name=name)

        images = os.listdir(path)
        for image_i in range(len(images)):
            imname = images[image_i]
            labelname = imname[:-3] + 'xml'
            impath = os.path.join(path, imname)
            labelpath = os.path.join(LABELS_DIR, labelname)
            self.add_image(source=SOURCE, image_id=image_i, path=impath,
                           labelpath=labelpath, )

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

        if self.end_to_end:
            score = AxeDataset.score_from_labelpath(labelpath)
            label = self.class_labels_to_ids[score]
        else:
            label = AXE_ID

        return (xmin, ymin, xmax, ymax), (height, width), label

    # def load_image(self, image_id):
    #     # no change from original load_image() function
    #     return super().load_image(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        bounds, imshape, label = self.get_label(info["labelpath"])
        xmin, ymin, xmax, ymax = bounds
        height, width = imshape

        # only one label to be applied
        mask = np.zeros(shape=(height, width, 1), dtype=np.uint8)
        mask[ymin:ymax, xmin:xmax, 0] = 1
        # np.ones([mask.shape[-1]], dtype=np.int32) * desired_id
        # only one axe per image
        return mask.astype(np.bool), np.array([label]).astype(np.int32)

    @staticmethod
    def score_from_labelpath(labelpath):
        return labelpath[-5]
