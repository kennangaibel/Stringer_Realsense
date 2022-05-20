import pyrealsense2 as rs
import numpy as np
import cv2
from PIL import Image
# THIS COMPILES
# 1. save_single_frameset

# !!: May have to do the save_single_framset, output bag file as a single method
# !!: Next method will do the config rs.config.enable_device_from_file(config, file_name)
# !!: Run the rest of the program, including the 3D world-space coordinate from there

# Obtains a bag file from a single frame taken by L515
def get_bag_file():
    # Obtains a bag file from a single frame taken by L515
    # try:
    pipeline = rs.pipeline()
    config = rs.config()
    # lower resolution for pipeline.start(config) to work
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    # config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 15)
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.rgb8, 15)
    # Align objects
    align_to = rs.stream.color
    align = rs.align(align_to)

    profile = pipeline.start(config)
    # Bag file representing single frame taken by camera
    path = rs.save_single_frameset()

    for x in range(100):
        pipeline.wait_for_frames()

    frame = pipeline.wait_for_frames()
    path.process(frame)

    # !: Must I use pipeline.stop?
    pipeline.stop()

    # Returns the single frame captured by the camera
    return path

# From get_corners.py
# Gets an array of pixels that represent corners of an image
def get_corner_pixels(path):
    # Creates Pipeline
    pipeline = rs.pipeline()
    # Creates a config object
    config = rs.config()
    # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
    # Allows us to use the bag file created by save_single_frameset
    rs.config.enable_device_from_file(config,'./test.bag')
    pipeline.start(config)

    # Converts bag file to numpy image that opencv can use
    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Gets numpy array color_image and converts it to a png that opencv can use
    im = Image.fromarray(color_image)
    # What to save the image as
    im.save('stringer_image.png')

    # Loads image and runs through opencv corner finding algorithm
    img = cv2.imread('stringer_image.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 5, 3, 0.04)
    ret, dst = cv2.threshold(dst, 0.1 * dst.max(), 255, 0)
    dst = np.uint8(dst)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)
    # !: Filter out corners based on DEPTH
    # Corner will always be a certain depth, only take the corner that is of that depth
        # If depthOf(corners[i]) = DEPTH_RANGE: print
    for i in range(1, len(corners)):
        print(corners[i])
    img[dst > 0.1 * dst.max()] = [0, 0, 255]
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows
    # Returns the array of corners from our program

    return corners

# # Converts 2D pixel coordinates to 3D world coordinates
# def deproject_pixels(corners):
#

path = get_bag_file()
get_corner_pixels(path)