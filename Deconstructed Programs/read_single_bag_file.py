import pyrealsense2 as rs
import numpy as np
import cv2

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

# !: APPLY FILTER TO GET RID OF HOLE ANOMALIES) ----------------------------------
# Fetch color and depth frames
depth_frame = frame.get_depth_frame()
color_frame = frame.get_color_frame()

#filter definition
dec_filter = rs.decimation_filter() # Decimation - reduces depth frame density
spat_filter = rs.spatial_filter() # Spatial - edge-preserving spatial smoothing
temp_filter = rs.temporal_filter() # Temporal - reduces temporal noise

depth_to_disparity = rs.disparity_transform(True)
disparity_to_depth = rs.disparity_transform(False)

# Using Filtering
frame = dec_filter.process(depth_frame)
spat_filter.set_option(rs.option.filter_magnitude, 4)
spat_filter.set_option(rs.option.holes_fill, 3)
frame = depth_to_disparity.process(frame)
frame = spat_filter.process(frame)
frame = temp_filter.process(frame)
frame = disparity_to_depth.process(frame)
# ----------------------------------------------Filter DONE
filt.process(frame)
# Returns the single frame captured by the camera
print("ROSBAG file is ", filt)
# except Exception as e:
#     print(e)
#     pass
