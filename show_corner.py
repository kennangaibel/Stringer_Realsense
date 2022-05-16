import pyrealsense2 as rs
import numpy as np
import cv2
from PIL import Image
# WORKING PROGRM THAT TAKES A RECORDED BAG FILE AND DETECTS CORNERS


# Creates Pipeline
pipeline = rs.pipeline()
# Creates a config object
config = rs.config()
# Tell config that we will use a recorded device from file to be used by the pipeline through playback.
# Allows us to use the bag file created by save_single_frameset
# !: Can test with Frameset 97.bag
rs.config.enable_device_from_file(config,'test.bag')
pipeline.start(config)

# Converts bag file to numpy image that opencv can use
# Wait for a coherent pair of frames: depth and color
frames = pipeline.wait_for_frames()
depth_frame = frames.get_depth_frame()
color_frame = frames.get_color_frame()

# Convert images to numpy arrays
depth_image = np.asanyarray(depth_frame.get_data())
color_image = np.asanyarray(color_frame.get_data())



# Gets numpy array color_image and converts it to a jpeg that opencv can use
im = Image.fromarray(color_image)
# What to save the image as
im.save('stringer_image.png')

# read the image
img = cv2.imread('stringer_image.png')
# convert image to gray scale image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# detect corners with the goodFeaturesToTrack function.
# corners = cv2.goodFeaturesToTrack(gray, 27, 0.01, 10)
# corners = np.int0(corners)
gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)

# result is dilated for marking the corners, not important
dst = cv2.dilate(dst, None)
# Threshold for an optimal value, it may vary depending on the image.
img[dst > 0.01 * dst.max()] = [0, 0, 255]
cv2.imshow('dst', img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()