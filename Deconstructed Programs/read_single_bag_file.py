import pyrealsense2 as rs
import numpy as np
import cv2

# Obtains a bag file from a single frame taken by L515
try:
    pipeline = rs.pipeline()
    config = rs.config()


    # lower resolution for pipeline.start(config) to work
    # 848x480 depth for 435
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    # config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 15)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 15)

    # Align objects
    align_to = rs.stream.color
    align = rs.align(align_to)

    profile = pipeline.start(config)

    filt = rs.save_single_frameset()

    for x in range(100):
        pipeline.wait_for_frames()

    frame = pipeline.wait_for_frames()

    # !: APPLY FILTER TO GET RID OF HOLE ANOMALIES) ----------------------------------
    # Fetch color and depth frames
    # Align the depth frame to color frame
    aligned_frames = align.process(frame)
    # Get aligned frames
    depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
    color_frame = aligned_frames.get_color_frame()
    # Spatial: Applies edge-preserving smoothing of depth data.
    # Spatial solves the 0 depth anomaly on edges
    spat_filter = rs.spatial_filter() # Spatial - edge-preserving spatial smoothing
    # Temporal: Filters depth data by looking into previous frames.
    # Uses previous frames to decide whether missing values should be filled
    # with previous data
    temp_filter = rs.temporal_filter() # Temporal - reduces temporal noise
    # Sets up custom parameters for spatial filter
    # Can play around with parameters to see what yields
    # best result
    # Number of filter iterations
    spat_filter.set_option(rs.option.filter_magnitude, 4)
    # The greater this option parameter is set, the larger
    # the holes are filled
    spat_filter.set_option(rs.option.holes_fill, 5)
    frame = spat_filter.process(frame)
    frame = temp_filter.process(frame)
    # ----------------------------------------------Filter DONE
    filt.process(frame)
    # Returns the single frame captured by the camera
    print("ROSBAG file is ", filt)
except Exception as e:
    print(e)
    pass
