# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 18:31:39 2021

@author: BW
"""
import cv2
import pyrealsense2 as rs
# from pupil_apriltags import Detector
import numpy as np

cv2.destroyAllWindows()
WINDOW_SCALE = 1
TAG_SIZE = 0.065  # meter
screenWidth = 640;  # pixel
screenHeight = 480;  # pixel


def low_pass(x, y):
    c1 = 0.003328
    c2 = 0.9967
    delta = 1 / x.shape[0] * ((c1 - 1) * np.sum(x, axis=0) + c2 * np.sum(y, axis=0))
    return delta


def get_rs_param(cfg):
    profile = cfg.get_stream(rs.stream.color)
    intr = profile.as_video_stream_profile().get_intrinsics()
    return [intr.fx, intr.fy, intr.ppx, intr.ppy]


def Sift_det(img, sift):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray, None)
    #   img=cv2.drawKeypoints(img,kp,img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return None, kp, des


def GFTT(img):
    # Detecting corners
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #   gray=cv2.convertScaleAbs(gray, beta=0, alpha=2)
    corners = np.int0(cv2.goodFeaturesToTrack(gray, maxCorners=20, qualityLevel=0.01, minDistance=10))
    return corners


def harris_feature(img, val):
    # Detector parameters
    blockSize = 2
    apertureSize = 3
    k = 0.04
    # Detecting corners
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #   gray=cv2.convertScaleAbs(gray, beta=0, alpha=2)
    dst = cv2.cornerHarris(gray, blockSize, apertureSize, k)
    # Normalizing
    dst_norm = np.empty(dst.shape, dtype=np.float32)
    cv2.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    #  dst_norm_scaled = cv2.convertScaleAbs(dst_norm)
    corners = np.where(dst_norm > val)
    for i in range(corners[0].shape[0]):
        gray = cv2.circle(gray, (corners[1][i], corners[0][i]), 5, (0), 2)
    cv2.imshow('Raw', gray)
    return corners


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, screenWidth, screenHeight, rs.format.bgr8, 30)

# Start streaming
cfg = pipeline.start(config)

# at_detector = Detector(families='tag36h11',
#                        nthreads=1,
#                        quad_decimate=1.0,
#                        quad_sigma=0.0,
#                        refine_edges=1,
#                        decode_sharpening=0.25,
#                        debug=0)
y_old = [None, None]
des = [None, None]
bf = cv2.BFMatcher()
sift = cv2.SIFT_create()
im = [None, None]
kp = [None, None]
init = False
shift = [0, 0]
try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        # Convert images to numpy arrays
        image = np.asanyarray(color_frame.get_data())
        im[1] = image

        # Tag detection
        _, kp[1], des[1] = Sift_det(image, sift)
        try:
            matches = bf.knnMatch(des[0], des[1], k=2)
            if init:
                good = [[m] for m, n in matches if m.distance < 0.5 * n.distance]
                #                    image = cv2.drawMatchesKnn(im[0],kp[0],im[1],kp[1],good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                good_kp = np.asarray([[kp[0][match[0].queryIdx].pt, kp[1][match[0].trainIdx].pt] for match in good])
                shift = low_pass(good_kp[:, 1, :], good_kp[:, 0, :])
        except:
            print("Not enough features!")
        # image=cv2.circle(image, (int(x[0]),int(x[1])), radius=5, color=[0,0,255], thickness=-1)
        trans = np.float32([[1, 0, int(shift[0])], [0, 1, int(shift[1])]])
        im[1] = cv2.warpAffine(im[1], trans, (im[1].shape[1], im[1].shape[0]))
        crop = 0
        cv2.imshow('Stabilized', im[1][crop:screenHeight - crop, crop:screenWidth - crop])
        cv2.imshow('Raw', image)
        im[0] = im[1]
        _, kp[0], des[0] = Sift_det(im[1], sift)

        init = True
        cv2.waitKey(1)
except KeyboardInterrupt:
    cv2.destroyAllWindows()
    pipeline.stop()
