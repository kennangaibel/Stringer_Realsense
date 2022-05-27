import pyrealsense2 as rs
import numpy as np
import cv2
from PIL import Image
from datetime import datetime

# THIS COMPILES
# 1. save_single_frameset
MIN_DEPTH = 0.4
MAX_DEPTH = 0.7

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
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 15)

    profile = pipeline.start(config)
    # Bag file representing single frame taken by camera
    path = rs.save_single_frameset()

    for x in range(100):
        pipeline.wait_for_frames()

    frame = pipeline.wait_for_frames()
    path.process(frame)

    # !: Must I use pipeline.stop?
    # pipeline.stop()

    # Returns the single frame captured by the camera
    return path

# From get_corners.py
# Gets an array of pixels that represent corners of an image
def get_corner_pixels():
    # Creates Pipeline
    pipeline = rs.pipeline()
    # Creates a config object
    config = rs.config()
    # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
    # Allows us to use the bag file created by save_single_frameset
    # rs.config.enable_device_from_file(config,'./test.bag')
    # !: Need to convert path into a string somehow

    # CAN JUST SKIP ALL OF THIS AND ONLY RUN get_corner_pixels
    rs.config.enable_device_from_file(config,'RealSense Frameset 117.bag')

    # rs.config.enable_device_from_file(config,path,True)

    profile = pipeline.start(config)
    # ADDED ALIGN DEPTH TO COLOR SINCE DIFFERENT RESOLUTIONS ------------------
    # Converts bag file to numpy image that opencv can use
    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    #  Aligns Depth->Color
    align = rs.align(align_to)


    # Align the depth frame to color frame
    aligned_frames = align.process(frames)

    # Get aligned frames
    depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
    color_frame = aligned_frames.get_color_frame()
    # DELETE IF DOESN"T WORK -------------------------------------------------------
    # depth_frame = frames.get_depth_frame()
    # color_frame = frames.get_color_frame()

    # !: Filter doesn't affect anything here, maybe need to do
    # it earlier?
    # Spatial: Applies edge-preserving smoothing of depth data.
    # Spatial solves the 0 depth anomaly on edges
    spat_filter = rs.spatial_filter()  # Spatial - edge-preserving spatial smoothing
    # Temporal: Filters depth data by looking into previous frames.
    # Uses previous frames to decide whether missing values should be filled
    # with previous data
    temp_filter = rs.temporal_filter()  # Temporal - reduces temporal noise
    # Sets up custom parameters for spatial filter
    # Can play around with parameters to see what yields
    # best result
    spat_filter.set_option(rs.option.filter_magnitude, 4)
    spat_filter.set_option(rs.option.holes_fill, 3)
    frame = spat_filter.process(frames)
    frame = temp_filter.process(frame)

    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Gets numpy array color_image and converts it to a png that opencv can use
    im = Image.fromarray(color_image)
    # What to save the image as
    # Adds date to png file of picture taken
    # date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
    # im.save(f"Stringer{date}.png")
    # img = cv2.imread(f"Stringer{date}.png")

    # !: In final implementation, delete png after use
    im.save('stringer_image.png')
    # Loads image and runs through opencv corner finding algorithm
    img = cv2.imread('stringer_image.png')
    # Gets rid of "salt and pepper" noise
    # img = cv2.medianBlur(img, 3)
    # 11, 21, 7
    img = cv2.bilateralFilter(img, 11, 21, 7)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 5, 3, 0.04)
    ret, dst = cv2.threshold(dst, 0.1 * dst.max(), 255, 0)
    dst = np.uint8(dst)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)
    # Filter out corners based on DEPTH, only want corner within certain depth range
    # List to store coordinates of corners we actually want
    filtered_corners = []
    for i in range(1, len(corners)):
        # Gets the pixel coordinates of the corner
        # x_pixel = int(corners[i][0] / 3)
        # y_pixel = int(corners[i][1] / 3)

        x_pixel = int(corners[i][0])
        y_pixel = int(corners[i][1])
        print(f"X pixel: {x_pixel}")
        print(f"Y pixel: {y_pixel}")

        # Filters based on desired depth range
        # !: Reason this doesn't work is because depth it is
        # getting is 0. This should be fixed with filtering
        # if (depth_image[y_pixel][x_pixel] > MIN_DEPTH and
        #         depth_image[y_pixel][x_pixel] < MAX_DEPTH):
        print("Corner detected:")
        print(corners[i])
        # print("depth x,y")
        # print(depth_image[x_pixel][y_pixel])
        print("depth y,x")
        print(depth_image[y_pixel][x_pixel])
        # !: is depth array y then x or x then y?
        if ((MIN_DEPTH < depth_frame.get_distance(int(corners[i][0]), int(corners[i][1])) < MAX_DEPTH)):
            filtered_corners.append(corners[i])
            print("filtered corners")
            print(corners[i])
            print("depth")
            print(depth_image[y_pixel][x_pixel])
            print(depth_frame.get_distance(int(corners[i][0]), int(corners[i][1])))
    img[dst > 0.1 * dst.max()] = [0, 0, 255]

    # Deproject pixels from filtered_corners
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    depth_min = MIN_DEPTH  # meter
    depth_max = MAX_DEPTH  # meter

    depth_intrin = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    color_intrin = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    depth_to_color_extrin = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_extrinsics_to(
        profile.get_stream(rs.stream.color))
    color_to_depth_extrin = profile.get_stream(rs.stream.color).as_video_stream_profile().get_extrinsics_to(
        profile.get_stream(rs.stream.depth))
    print("filtered_corners: ")
    print(filtered_corners)
    # List to store the real world coordinates of filtered_corners
    coordinate = []
    # !: CHECK THAT FILTERED_CORNERS IS SAME TYPE/FORMAT AS COLOR POINT
    for color_point in filtered_corners:
        # get the 3D coordinate
        # camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrs, [x, y], dis)
        print(color_point)
        # depth_pixel = rs.rs2_project_color_pixel_to_depth_pixel(
        #     depth_frame.get_data(), depth_scale,
        #     depth_min, depth_max,
        #     depth_intrin, color_intrin, depth_to_color_extrin, color_to_depth_extrin, color_point)
        # print(depth_pixel)
        # Get the  depth value
        # depth = depth_frame.get_distance(int(depth_pixel[0]), int(depth_pixel[1]))
        # depth = depth_frame.get_distance(color_point[0], color_point[1])
        # # depth = depth_image[filtered_corners[1]][filtered_corners[0]]
        # # depth_point_ = rs.rs2_deproject_pixel_to_point(depth_intrin, [depth_pixel[0], depth_pixel[1]], depth)
        # depth_point_ = rs.rs2_deproject_pixel_to_point(depth_intrin, color_point[0], color_point[1], depth)

        depth = depth_frame.get_distance(int(color_point[0]), int(color_point[1]))
        # depth = depth_frame.get_distance(color_point[0], color_point[1])
        depth_point_ = rs.rs2_deproject_pixel_to_point(depth_intrin, [color_point[0], color_point[1]], depth)

        # da
        print("depth value ")
        print(depth_point_[0])
        print(depth_point_[1])
        print(depth_point_[2])

        coordinate.append(depth_point_)
    # Shows image with corners until any key press
    # (Delete later for program to run without interruption)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows

    # Returns the array of corners from our program
    return coordinate

# Converts 2D pixel coordinates to 3D world coordinates
# def deproject_pixels(corners):
# path = get_bag_file()
coordinate = get_corner_pixels()
exit()