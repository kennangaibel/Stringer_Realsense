import pyrealsense2 as rs
import numpy as np
import cv2

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
    filt = rs.save_single_frameset()

    for x in range(100):
        pipeline.wait_for_frames()

    frame = pipeline.wait_for_frames()
    filt.process(frame)

    # !: Must I use pipeline.stop?
    # pipeline.stop()

    # Returns the single frame captured by the camera
    return filt

# !: Is this the solution to my problems to enter opencv?
    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())



# Finds coordinate
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

    # Configure the pipeline to stream the depth stream
    # Change this parameters according to the recorded bag file resolution
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 15)
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.rgb8, 15)

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

path = get_bag_file()
find_corner(path)
