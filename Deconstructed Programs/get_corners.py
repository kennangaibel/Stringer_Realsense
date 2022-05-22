import pyrealsense2 as rs
import numpy as np
import cv2
from PIL import Image

MIN_DEPTH = 0.4
MAX_DEPTH = 1.0

# WORKING PROGRAM THAT TAKES A RECORDED BAG FILE AND DETECTS CORNERS

# Creates Pipeline
pipeline = rs.pipeline()
# Creates a config object
config = rs.config()
# Tell config that we will use a recorded device from file to be used by the pipeline through playback.
# Allows us to use the bag file created by save_single_frameset
rs.config.enable_device_from_file(config,'./test.bag')
pipeline.start(config)

# Wait for a coherent pair of frames: depth and color
frames = pipeline.wait_for_frames()
# Converts bag file to numpy of depth data
depth_frame = frames.get_depth_frame()
# Converts bag file to numpy image that opencv can use
color_frame = frames.get_color_frame()

# Convert images to numpy arrays that opencv can use
depth_image = np.asanyarray(depth_frame.get_data())
color_image = np.asanyarray(color_frame.get_data())

# Gets numpy array color_image and converts it to a png that opencv can use
im = Image.fromarray(color_image)
# What to save the image as
im.save('stringer_image.png')

# Loads image and runs opencv corner detection algorithm
img = cv2.imread('stringer_image.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv2.cornerHarris(gray,5,3,0.04)
ret, dst = cv2.threshold(dst,0.1*dst.max(),255,0)
dst = np.uint8(dst)
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
# Prints out pixel coordinates of all detected corners

# List to store coordinates of corners we actually want
filtered_corners = []
# Depth resolution: 640 x 480
# RGB resolution: 1920 x 1080
# Solution: Divide the depth pixel value, so 1920 / 640 = 3
for i in range(1, len(corners)):
    # Gets the pixel coordinates of the corner
    x_pixel = int(corners[i][0] / 3)
    y_pixel = int(corners[i][1] / 3)
    # print(x_pixel)
    # print(y_pixel)
    # If the corner is in depth range, add to filtered_corners
    # if (depth_image[y_pixel][x_pixel] > MIN_DEPTH and
    #     depth_image[y_pixel][x_pixel] < MAX_DEPTH):
    if (depth_image[y_pixel][x_pixel] < 1):
        filtered_corners.append(corners[i])
        print(corners[i])
print(filtered_corners)
# for i in range(1, len(corners)):
#     print(corners[i])

img[dst>0.1*dst.max()]=[0,0,255]
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows
print(1)