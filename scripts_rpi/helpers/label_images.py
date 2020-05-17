import cv2
import os
import sys

# folder_path = "~/Documents/axe_images/positives"
ignore_path = ".DS_Store"

def main():
    folder_path = sys.argv[1]
    for img_file in os.listdir(folder_path):
        if img_file == ignore_path: continue
        old_path = os.path.join(folder_path, img_file)
        img = cv2.imread(old_path)
        cv2.imshow('image',img)
        print("Enter score:")
        score_ascii = cv2.waitKey(0)
        score = int(chr(score_ascii % 256))
        print(score)
        cv2.destroyAllWindows()

        img_name = img_file[:-4]
        new_path = os.path.join(folder_path,
            img_name + ("_score:%d" % score) + ".png")
        os.rename(old_path, new_path)

main()
