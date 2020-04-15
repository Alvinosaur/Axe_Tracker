import cv2
import os
import xml.etree.ElementTree

positives_dir = "../../../axe_images/positives"
labels_dir = "../../../axe_images/annotated_positives"

new_positives_dir = "../../../axe_images/new_positives"
new_labels_dir = "../../../axe_images/new_annotated"

FINAL_SIZE = 128
SCALE = 0.25  # 512 -> 128 size images
orig_h, orig_w = 700, 600
target_center = (357, 347)  # (row, col), found manually
orig_center = (350, 300)
dcy, dcx = target_center[0] - orig_center[0], 0
target_h, target_w = 512, 512


def main():
    img_names = os.listdir(positives_dir)
    cy, cx = target_center
    dy, dx = target_h//2, target_w//2
    top, bot, left, right = cy-dy, cy+dy, cx-dx, cx+dx
    label_shift_h, label_shift_w = top - 0, left - 0
    for img_name in img_names:
        if img_name == '.DS_Store': continue
        et, in_bounds = offset_label(img_name[:-3] + 'xml', label_shift_h, 
            label_shift_w)
        
        # if shifted label positions still within image bounds, add to positives
        if in_bounds:
            img = resize_img(img_name, top, bot, left, right)

            # write new label
            new_label_path = os.path.join(new_labels_dir, img_name[:-3] + 'xml')
            et.write(new_label_path)

            # write new image
            new_img_path = os.path.join(new_positives_dir, img_name)
            cv2.imwrite(new_img_path, img)


def resize_img(file, top, bot, left, right):
    # crop to be 512 x 512, centered around target
    img = cv2.imread(filename=os.path.join(positives_dir, file))
    img = img[top:bot, left:right]

    # scale down to 128 x 128
    img = cv2.resize(img, dsize=(FINAL_SIZE, FINAL_SIZE))
    # if image still contains axe, then keep as new positive, else don't add
    return img


def offset_label(file, shift_h, shift_w):
    # Open original file
    old_path = os.path.join(labels_dir, file)
    et = xml.etree.ElementTree.parse(old_path)
    root = et.getroot()
    
    # update image size
    size = root.find('size')
    width = size.find('width')
    height = size.find('height')
    width.text = str(FINAL_SIZE)
    height.text = str(FINAL_SIZE)
    
    obj = root.find('object')
    bbox = obj.find('bndbox')
    xmin, xmax = bbox.find('xmin'), bbox.find('xmax')
    ymin, ymax = bbox.find('ymin'), bbox.find('ymax')

    xmin_val = max(0, int((int(xmin.text) - shift_w) * SCALE))
    xmax_val = min(FINAL_SIZE, int((int(xmax.text) - shift_w) * SCALE))
    ymin_val = max(0, int((int(ymin.text) - shift_h + dcy) * SCALE))
    ymax_val = min(FINAL_SIZE, int((int(ymax.text) - shift_h + dcy) * SCALE))

    xmin.text = str(xmin_val)
    xmax.text = str(xmax_val)
    ymin.text = str(ymin_val)
    ymax.text = str(ymax_val)

    in_bounds = (
        (0 <= xmin_val <= FINAL_SIZE) and (0 <= xmax_val <= FINAL_SIZE) and
        (0 <= ymin_val <= FINAL_SIZE) and (0 <= ymax_val <= FINAL_SIZE)
    )

    return et, in_bounds

main()