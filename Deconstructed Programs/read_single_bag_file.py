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
filt.process(frame)
# Returns the single frame captured by the camera
print("ROSBAG file is ", filt)
# except Exception as e:
#     print(e)
#     pass
