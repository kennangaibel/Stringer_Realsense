import pyrealsense2 as rs
import bagpy
import rosbag
from bagpy import bagreader
import cv2
from cv_bridge import CvBridge
import numpy as np

# # Gets Bag file "filt"
# pipeline = rs.pipeline()
# config = rs.config()
# # lower resolution for pipeline.start(config) to work
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# # config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 15)
# config.enable_stream(rs.stream.color, 1920, 1080, rs.format.rgb8, 15)
# # Align objects
# align_to = rs.stream.color
# align = rs.align(align_to)
#
# profile = pipeline.start(config)
# filt = rs.save_single_frameset()
#
# for x in range(100):
#     pipeline.wait_for_frames()
#
# frame = pipeline.wait_for_frames()
# filt.process(frame)

# Converts Bag File to PNG
# Replace BAGFILE with path/filt
# ROOT_DIR = 'C:\Users\kenna\PycharmProjects\Wong_program'
BAGFILE = 'test.bag'

# bag = bagreader('test.bag')
bag = rosbag.Bag('test.bag')
for i in range(2):
    if (i == 0):
        TOPIC = '/camera/depth/image_rect_raw'
        DESCRIPTION = 'depth_'
    else:
        TOPIC = '/camera/color/image_raw'
        DESCRIPTION = 'color_'
    # !: find what is bag.read_messages and how to replace
    image_topic = bag.read_messages(TOPIC)
    for k, b in enumerate(image_topic):
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(b.message, b.message.encoding)
        cv_image.astype(np.uint8)
        # Outputs out png of depth
        if (DESCRIPTION == 'depth_'):
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(cv_image, alpha=0.03), cv2.COLORMAP_JET)
            # Saves image to given directory
            cv2.imwrite(r"C:\Users\kenna\PycharmProjects\Wong_program" + '/depth/' + DESCRIPTION + str(b.timestamp) + '.png', cv_image)
            print("done1")
        else:
            # Outputs png of color
            # cv2.imwrite(r"C:\Users\kenna\PycharmProjects\Wong_program" + '/color/' + DESCRIPTION + str(b.timestamp) + '.png', cv_image)
            cv2.imwrite(r"C:\Users\kenna\PycharmProjects\Wong_program")
            print("done2")
        print('saved: ' + DESCRIPTION + str(b.timestamp) + '.png')


bag.close()

print('PROCESS COMPLETE')