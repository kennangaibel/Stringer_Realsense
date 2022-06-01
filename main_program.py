import pyrealsense2 as rs
import numpy as np
import cv2
from PIL import Image

MIN_DEPTH = 0.4
MAX_DEPTH = 0.7

# Gets an array of pixels that represent corners of an image
def get_corner_pixels():
    try:
        # Creates Pipeline
        pipeline = rs.pipeline()
        # Creates a config object
        config = rs.config()
        # Set resolutions for color and depth frames
        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 15)

        # Replays an already captured bag file rather than using a present stream
        # Comment out rs.config.enable_device_from_file if you want to take a picture
        # in the present and automate the process
        rs.config.enable_device_from_file(config,'RealSense Frameset 117.bag')

        profile = pipeline.start(config)

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        # Creates an align object
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        #  Aligns depth frame to color frame
        align = rs.align(align_to)
        aligned_frames = align.process(frames)

        # Get aligned frames
        depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        spat_filter = rs.spatial_filter()  # Spatial - edge-preserving spatial smoothing
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
        # Runs opencv algorithm
        corners = find_corners(color_image)

        # Filter out corners based on DEPTH, only want corner within certain depth range
        # List to store coordinates of corners we actually want
        filtered_corners = []
        for i in range(1, len(corners)):
            # Gets the pixel coordinates of the corner
            x_pixel = int(corners[i][0])
            y_pixel = int(corners[i][1])
            # print(f"X pixel: {x_pixel}")
            # print(f"Y pixel: {y_pixel}")

            # Filters based on desired depth range
            print("Corner detected:")
            print(corners[i])
            # Applies depth filter
            if ((MIN_DEPTH < depth_frame.get_distance(x_pixel, y_pixel) < MAX_DEPTH)):
                # Adds the corner that gets through depth filter to a list
                filtered_corners.append(corners[i])
                print(f"Filtered Corner Pixel Coordinate: {corners[i]}")
                filtered_corner_depth = depth_frame.get_distance(int(corners[i][0]), int(corners[i][1]))
                print(f"Filtered Corner Pixel Depth: {filtered_corner_depth}\n")

        # Deproject pixels from filtered_corners
        depth_intrin = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

        print("filtered_corners list: ")
        print(filtered_corners)
        # List to store the real world coordinates of filtered_corners
        coordinates = []
        # Converts all pixels into (x,y,z) coordinates
        for color_pixel in filtered_corners:
            # get the 3D coordinate
            print(f"RGB Pixel Coordinate: {color_pixel}")
            # Gets depth value of a pixel
            depth = depth_frame.get_distance(int(color_pixel[0]), int(color_pixel[1]))
            # Gets the 3D coordinate of that pixel
            depth_point_ = rs.rs2_deproject_pixel_to_point(depth_intrin, [color_pixel[0], color_pixel[1]], depth)

            x = depth_point_[0]
            y = depth_point_[1]
            z = depth_point_[2]

            print(f"3D Coordinate: ({x}, {y}, {z})\n")

            coordinates.append(depth_point_)

        # Returns the array of corner coordinates from our program
        return coordinates
    # Prints an error if something goes wrong
    except Exception as e:
        print(e)
        exit()

def find_corners(color_image):
    # Gets numpy array color_image and converts it to a png that opencv can use
    im = Image.fromarray(color_image)

    # !: In final implementation, delete png after use
    im.save('stringer_image.png')
    # Loads image and runs through opencv corner finding algorithm
    img = cv2.imread('stringer_image.png')
    # Gets rid of "salt and pepper" noise
    img = cv2.bilateralFilter(img, 11, 21, 7)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 5, 3, 0.04)
    ret, dst = cv2.threshold(dst, 0.1 * dst.max(), 255, 0)
    dst = np.uint8(dst)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)
    img[dst > 0.1 * dst.max()] = [0, 0, 255]
    # Shows the image with the algorithm
    cv2.imshow('image', img)
    return corners

# Array of 3D real-world coordinates from corners
coordinate = get_corner_pixels()
# Shows image with corners until any key press
# (Delete later for program to run without interruption)
cv2.waitKey(0)
cv2.destroyAllWindows
exit()