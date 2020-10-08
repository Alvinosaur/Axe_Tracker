# Axe_Tracker: Automated Scoring for Axe Throwing Competition
## Brief Summary
Axe-throwing involves throwing an axe at a bulls-eye targt and scoring the throw based on the innermost ring that the axe tip is embedded in. Automating this process thus involves three steps:
1. Detect Axe
2. Segment Axe tip(some bounding box)
3. Segment Bullseye rings(circles of varying radii with same center)
4. Determine innermost ring that axe tip lies in

To detect axes, we compare two image frames, one with and one without an axe, using the SSIM algorithm. Details for detection can be found in the classical approach google slides. Axe detection is also used for automatic data collection on the site: a continuously running script takes images at a given frequency and saves images when an axe is detected.

To segment the axe tip, we use MaskRCNN, which performs pixel-level segmentation. In this case, we only have two classes: background or axe tip. The most important hyperparameters include TRAIN_ROIS_PER_IMAGE (set to small value of 32) and MAX_GT_INSTANCES (set to 1) since there is only ever one axe present in an image. Although positives and negatives are collected automatically using axe detection, we need to manually label the images. I used [RectLabel's free trial](https://apps.apple.com/us/app/rectlabel-for-object-detection/id1210181730?mt=12) to label pixels (simple bounding boxes). I also used the *label_images.py* script described below to quickly hand-label the score of each axe throw. 

## Performance
Hardware used is a Raspberry Pi 4 Model B.

Runtime for axe detection is 1.204s
Ruuntime for axe scoring is 11.04s. 
End-to-end performs much more poorly because of lack enough training data.
Training size: 269 images
Val size: 59
Test size: 58

### Final Performance
![final_performance_comparison.jpg](https://github.com/Alvinosaur/Axe_Tracker/blob/master/final_performance_comparison.jpg?raw=true)

### Example Segmentation
![example_segmentation.png](https://github.com/Alvinosaur/Axe_Tracker/blob/master/example_segmentation.png?raw=true)

### Example Ring Detection
![example_ring_detection.jpg](https://github.com/Alvinosaur/Axe_Tracker/blob/master/example_ring_detection.jpg?raw=true)

### Train and Validation Loss Curves
![train_loss_curves.jpg](https://github.com/Alvinosaur/Axe_Tracker/blob/master/train_loss_curves.jpg?raw=true)

### Camera Mount Mechanical Design
![camera_mechanical_mount.jpg](https://github.com/Alvinosaur/Axe_Tracker/blob/master/camera_mechanical_mount.jpg?raw=true)

## Folder Layout
### All relevant scripts can be found in the *scripts_rpi* folder. Within that folder are following:
- *helpers*
   - *collect_train_imgs.py*: automatically collect new training image pairs (with and without axe). The without-axe images aren't needed anymore though for the MaskRCNN approach since no image subtraction is performed.
   - *file_helpers.py*: basic IO and general string-parsing helpers
   - *label_images.py*: command-line interface for you as user to label new training images. You manually type the label of each image. 
   - *ransac_circle.py*: low-level helper used for getting radius and center of each ring of the bullseye target
   - *ring_detect.py*: high-level ring-detection module that uses the above ransac circle detector
   - *resize_relabel_images.py*: MaskRCNN takes images of size (128 x 128) so raw images need to be cropped and shrunk downn
   - *rgb_filter_calibration.py*: not really used, I use Matlab's color segmentation tool. Helps calibrate color filters for ring detection
   - *split_train_val_test.py*: copies over images from one directory into separate train, val, test folders. You can specify in the file the what proportion should be allocated to each.
- *maskrcnn_approach*: best-performing method
   - *AxeConfig.py*: describe config parameters of MaskRCNN
   - *AxeDataset.py*: dataset class used for loading all necessary images for train, val, or test
   - *AxeDetectModel.py*: the main prediction class 
   - *maskrcnnn_training.ipynb*: notebook for training the model. You should use the version in the Google Drive folder (link provided below)
- *neural_net_approach*: naive attempt at using simple MLP, left for reference but unused
- *params*: contains a yaml file describing all relevant parameters
- *classical_approach*: very first attempt using non-learned approaches. Serves as a good baseline for comparison
- *random_files*: old, unused files from first iteration of project, left for reference

- *benchmark_detection.py*: measures time for detecting(NOT SCORING) an axe. Detection simply involves subtracting two images and does not use MaskRCNN
- *benchmark_scoring.py*: measure time for scoring an axe throw. This uses MaskRCNN. 

### All training data can be found either in *axe_images* folder or in the Google Drive foler (link provided below)

## Other Important Links
- [Drive foler with all images and untouched training notebook](https://drive.google.com/drive/folders/1i-IyCfMGqFx7QGnfqr1J7kUjCxeZ2bJV?usp=sharing)
- [Slides with all progress updates + results(runtime estimates, accuracy of model)](https://docs.google.com/presentation/d/1uH9LAQfr25p53zjWGmO-NtlNtRkCWy8wm5Qc7UlbyAk/edit?usp=sharing)
- [Slides for classical approach for reference](https://docs.google.com/presentation/d/1Z8DYN7VcDcc7ToR8XDTzJtpuj3n0Rf8O7oLZPMprWzc/edit?usp=sharing)
- [Mechanical Mount Design](https://docs.google.com/presentation/d/1XZY_rIJcyj0cT6RWiLMSZJaU2oba6PDZcBjUC_9ZC8g/edit?usp=sharing)

## Thanks
I would like to thank [Dr. George Kantor](https://www.ri.cmu.edu/ri-faculty/george-a-kantor/) for mentoring me for this project. I would also like to thank Corey Deasy from [Lumberjaxes](https://axethrowingpgh.com/) for sponsoring this project. 

## Issues:
1. There's a chance that the weights .h5 file is incorrectly stored in github.
   In that case, just download from drive (maskrcnn_final_weights.h5) directly and scp over to the raspberry pi.

2. For different versions of OpenCV, cv2.findContours may either return 2 or 3
   values. See https://github.com/facebookresearch/maskrcnn-benchmark/issues/339

3. I used Tensorflow version 1.14.0 or 1.15.2, not the newer versions of 2.1
4. Keras version is 2.3.1

5. I followed this tutorial for installing tensorflow dependencies, but for the final
   installation of TF itself, I just used pip3 install tensorflow, which by
   default installed 1.14.0 and worked
   https://qengineering.eu/install-tensorflow-2.1.0-on-raspberry-pi-4.html

6. This tutorial  for raspberry pi ssh headless worked perfectly:
   https://desertbot.io/blog/headless-raspberry-pi-4-ssh-wifi-setup
