import cv2
import argparse
import numpy as np

def nothing(temp):
    return

def run_calibration_gui():
    """
    Purpose is to have vision tracking update color thresholds to match
    a player's puck color and lighting conditions. 
    Once finish calibrating, store settings in a json file.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--img1', dest='img1', required=True, 
                        help='path to first test image')
    args = parser.parse_args()
    img1 = cv2.imread(args.img1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    # Ordered from min to max HSV
    param_names = ['R_MIN', 'G_MIN', 'B_MIN', 'R_MAX', 'G_MAX', 'B_MAX']
    # maps threshold param from min to max
    param_bounds = {
        'R_MIN': (0, 255),  # (min, max)
        'G_MAX': (0, 255),
        'B_MIN': (0, 255),
        'R_MAX': (0, 255),
        'G_MIN': (0, 255),
        'B_MAX': (0, 255)
    }

    # create window to hold sliders
    cv2.namedWindow("Filter GUI")
    
    # create slider for min/max thresholds of HSV
    for param in param_names:
        cv2.createTrackbar(param, "Filter GUI", param_bounds[param][0], 
                            param_bounds[param][1], nothing)

    thresholds = dict()
    while True:
        # Handle user closing gui
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print(thresholds)
            exit()

        # record all the current values chosen by user
        
        temp = []
        for param_name in param_names:
            value = cv2.getTrackbarPos(param_name, "Filter GUI")
            # follow hsv guide, but scale to match opencv2's scale
            temp.append(value)

        print(temp)
        
        # 3 x 1 vec of min/max HSV
        thresholds['lower'] = temp[0:3]  # json-serializable list
        thresholds['upper'] = temp[3:6]
        lowerBound = np.array(thresholds['lower'], int)
        upperBound = np.array(thresholds['upper'], int)

        image_filter = cv2.inRange(img1, lowerBound, upperBound)

        # apply user-chosen thresholds to filter image
        new_image = cv2.bitwise_and(img1, img1, mask = image_filter)
        cv2.imshow("New Image", new_image)

run_calibration_gui()