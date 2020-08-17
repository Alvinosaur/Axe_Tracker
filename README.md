# Axe_Tracker
Automated Scoring for Axe Throwing Competition

Training workspace: https://drive.google.com/drive/folders/1i-IyCfMGqFx7QGnfqr1J7kUjCxeZ2bJV?usp=sharing

Issues:

1. There's a chance that the weights .h5 file is incorrectly stored in github.
   In that case, just download from drive directly and scp over.

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
