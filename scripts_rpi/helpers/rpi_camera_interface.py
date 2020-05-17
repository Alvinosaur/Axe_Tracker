import time         # tools to manage time (e.g., sleep)
import picamera     # tools for the Raspberry Pi camera

# set the camera up
def get_rpi_cam(w, h):
    camera = picamera.PiCamera()
    camera.resolution = (w,  h)
    camera.framerate = 24
    time.sleep(2) # sleep for 2 seconds to initialize camera hardware
    return camera
