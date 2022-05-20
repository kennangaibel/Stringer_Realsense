import pyrealsense2 as rs
import numpy as np
import cv2
from PIL import Image

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
def get_corner_pixels():
    # Creates Pipeline
    pipeline = rs.pipeline()
    # Creates a config object
    config = rs.config()
    # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
    # Allows us to use the bag file created by save_single_frameset
    # !: Can test with Frameset 97.bag
    rs.config.enable_device_from_file(config, './test.bag')
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

    # read the image
    # load image
    # Read image
    img = cv2.imread('stringer_image.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 5, 3, 0.04)
    ret, dst = cv2.threshold(dst, 0.1 * dst.max(), 255, 0)
    dst = np.uint8(dst)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)
    for i in range(1, len(corners)):
        print(corners[i])
    img[dst > 0.1 * dst.max()] = [0, 0, 255]
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows









def get_corner_pixel(color_image):
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

# Finds corner
def find_corner(path):
    # # Gets bag file from picture just taken
    # path = get_bag_file(self)

    # Creates Pipeline
    pipeline = rs.pipeline()
    # Creates a config object
    config = rs.config()
    # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
    # Allows us to use the bag file created by save_single_frameset
    # !: Can test with Frameset 97.bag
    rs.config.enable_device_from_file(config,path)

# Converts bag file to numpy image that opencv can use
    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    # Uses np array of colors to convert to corner image
    get_corner_pixel(color_image)


    # Start streaming from file
    profile = pipeline.start(config)

    # 2. Obtaining 3D world-space coordinate of a single specific pixel coordinate on an image -------------------------------
    # rs2_project_color_pixel_to_depth_pixel
    # There values are needed to calculate the mapping
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    depth_min = 0.11 #meter
    depth_max = 1.0 #meter

    depth_intrin = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    color_intrin = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    depth_to_color_extrin =  profile.get_stream(rs.stream.depth).as_video_stream_profile().get_extrinsics_to( profile.get_stream(rs.stream.color))
    color_to_depth_extrin =  profile.get_stream(rs.stream.color).as_video_stream_profile().get_extrinsics_to( profile.get_stream(rs.stream.depth))

    # !: What is color points?
    # a: the 4 colored dots -> replace with one corner dot
    color_points = [
        [400.0, 150.0],
        [560.0, 150.0],
        [560.0, 260.0],
        [400.0, 260.0]
    ]
    for color_point in color_points:
       #  !: Depth_point is a depth pixel?

       #  convert a color pixel to a depth pixel
       depth_point_ = rs.rs2_project_color_pixel_to_depth_pixel(
                    depth_frame.get_data(), depth_scale,
                    depth_min, depth_max,
                    depth_intrin, color_intrin, depth_to_color_extrin, color_to_depth_extrin, color_point)
       # Check if depth_point is the value that I want
       print(depth_point_)

# Gets bag file
path = get_bag_file()
# Finds corner of image from bag file
find_corner()
# Gets coordinate from pixel
get_coordinate(path)
