# Stringer_Realsense
Uses an Intel Realsense D435 for precise object detection. Detects the corner of a stringer and maps the pixel location into real-world 3D space.

# Specs/Suggestions

Make sure you are using python version 3.6-3.9. The realsense library will not work with version 3.10.
With the D435, optimal resolutions for aligning depth with RGB is setting depth resolution to 848x480 and RGB resolution to 1280x720.

**Deconstructed Programs** contains smaller programs that accomplish tasks that are good for testing the main program `main.py`. Most notably,
`read_single_bag_file.py` which will take a picture and output a bag file that you can save so that you can analyze the RGB and depth frames on Intel.Realsense.Viewer.exe while also testing it on the main program. `cvTest.py` is the other program that you can input image files into and run the computer vision algorithm. This is a good place to test and refine the computer vision algorithm based on your application.

# Retrieving the data

The camera takes a picture, retrieving a bag file that has both RGB and depth frames. Since these two frames are different resolutions, an align object `align` is created to align the depth frame to the color frame. Once aligned, the RGB and Depth frames are stored into numpy arrays `depth_image` and `color_image`. 

Note that the bag file does not need to be directly converted into a png for the computer vision algorithm, the RGB numpy array can easily be saved into an image file via `im = Image.fromarray(color_image)` where `color_image` is the RGB numpy array.

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
![image](https://user-images.githubusercontent.com/86447811/171096614-a89233b2-7610-40a9-bdd4-0c40ac79e9a9.png)
(Example of computer vision algorithm run on a piece of paper mimicking the stringer, red dots indicate detected corners)


**Mapping RGB Pixels to real-world 3D coordinates**

The corner locations found are stored in a list of pixel values `corners`, which then are filtered out based on desired depth into list `filtered corners`. For example, if we know the stringer (or desired object) is gonna be 1-1.2 meters away, we can filter based off that, and set `MIN_DEPTH = 1` and `MAX_DEPTH = 1.2`. This will get rid of corners detected in the background or foreground that we do not care about.

Finally, the pixels from these filtered corners are mapped into 3D real-world coordinates with the `rs2_deproject_pixel_to_point` function. Only one point is needed to map into 3D real-world space.

# Quirks to be aware of
Aligning depth frames to color frames `align_to = rs.stream.color`, will cause both frames to have the resolution of the color frame. This works the same way if you align color frames to depth frames via `align_to = rs.stream.depth`.

The depth of any pixel can be accessed via depth numpy array `depth_image`, however, note that if you want the depth of pixel (x, y) you will have to input `depth_image[y][x]` not `depth_image[x][y]` (OpenCV notation). Furthermore, accessing depth this way will provide depth in millimeters. On the other hand, depth can also be accessed through the function `depth_frame.get_distance(x, y)`, which not only uses the conventional coordinate notation as parameters, but also outputs the depth in meters. For this program, `depth_frame.get_distance(x, y) is used to minimize confusion.

Depending on your application the blurring in the OpenCV algorithm may block out too much noise or not enough. For the paper example shown above, I recommend filtering via `img = cv2.bilateralFilter(img, 11, 21, 7)`. However, for the actual stringer, a more intense blur from `img = cv2.medianBlur(img, 9)` is a lot more effective. I highly recommend taking a picture of whatever application you are using, saving it as an image, and running it through `cvTest.py`, messing with different blurring methods until you get the desired result. It doesn't have to be that perfect because the depth filter will take care of almost all noise with a narrow enough depth range.

Finally, avoid running any of the programs with Intel.Realsense.Viewer.exe on concurrently. It tends to cause confusing errors that stop any of the programs from working.
