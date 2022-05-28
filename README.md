# Stringer_Realsense
Uses an Intel Realsense D435 for precise object detection. Detects the corner of a stringer and maps the pixel location into real-world 3D space.

**Retrieving the data**

The camera takes a picture, retrieving a bag file. With the D435, optimal resolutions for aligning depth with RGB is setting depth resolution to 848x480 and RGB resolution to 1280x720. Since these two images are different resolutions, an align object `align` is created to align the depth frame to the color frame. Once aligned, the RGB and Depth frames are stored into numpy arrays `depth_image` and `color_image`.

**Filtering**

The image is then filtered through a spatial and temporal filter. This is to combat an "anomaly" in the depth data in which edges and random areas of the image were mapped to have a depth value of 0.

![image](https://user-images.githubusercontent.com/86447811/170658290-0cc7bb66-e418-4de8-aeb1-aaf6024bca83.png)

(The black edges and dots indicate a depth value of 0)

The spatial filter applies edge-preserving smoothing of depth data while the temporal filter uses previous frames to decide whether missing values should be filled with previous data.

**OpenCV Algorithm**

The RGB data in the form of a numpy array, `color_image`, is then saved as a png and inputted into an opencv algorithm:
1. Uses bilateral filtering to preserve edges but remove "salt and pepper" noise from the image so that camera can detect corners of concrete objects.
2. Runs a Harris Corner (corner detecting) algorithm
3. Function `cornerSubPix()` refines the corner location, finding the sub-pixel accurate location of corners or radial saddle points.



**Mapping RGB Pixel to real-word 3D coordinate**

The corner locations found are stored in a list of pixel values `corners`, which then are filtered out based on desired depth into list `filtered corneers`. For example, if we know the stringer (or desired object) is gonna be 1-1.2 meters away, we can filter based off that, and set `MIN_DEPTH = 1` and `MAX_DEPTH = 1.2`. This will get rid of corners detected in the background or foreground that we do not care about.

Finally, the pixels from these filtered corners are mapped into 3D real world coordinates with the `rs2_deproject_pixel_to_point` function.
