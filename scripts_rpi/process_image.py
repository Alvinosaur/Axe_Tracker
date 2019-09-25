
import skimage as ski  # scikit-image (for processing images)
import imutils
import cv2		# opencv library


def get_diff_SSIM(img1, img2, width, height):
    # crop image to only show target
    inc_w, inc_h = width // 4,  height // 4
    raw_img1 = raw_img1[inc_w:3*inc_w, inc_h:3*inc_h]
    raw_img2 = raw_img2[inc_w:3*inc_w, inc_h:3*inc_h]

    # convert the images to grayscale
    gray_img1 = cv2.cvtColor(raw_img1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(raw_img2, cv2.COLOR_BGR2GRAY)

    # compute Structural Similarity Index of the two images
    score = ski.measure.compare_ssim(gray_img1, gray_img2)
    return score
