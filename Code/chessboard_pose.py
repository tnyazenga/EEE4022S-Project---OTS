#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 18/12/14
@author: Sammy Pfeiffer
From http://docs.opencv.org/trunk/doc/py_tutorials/py_calib3d/py_calibration/py_calibration.html#calibration
Extra info:
http://docs.opencv.org/2.4.10/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html?highlight=calibratecamera#cv2.calibrateCamera
I love this guy "OpenCV error messages suck":
https://adventuresandwhathaveyou.wordpress.com/2014/03/14/opencv-error-messages-suck/
Get pose:
http://docs.opencv.org/trunk/doc/py_tutorials/py_calib3d/py_pose/py_pose.html
"""
import numpy as np
import cv2
import glob
import math
import pickle
import yaml

def _pdist(p1, p2):
    """
    Distance bwt two points. p1 = (x, y), p2 = (x, y)
    """
    return math.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2))

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

print("Initializing")
video_device_number = 0
camera = cv2.VideoCapture(video_device_number)
_, img = camera.read() #need black and white images
image_data = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #This converts the image to black and white
width, height = image_data.shape

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

n_rows = 8
n_cols = 12
n_cols_and_rows = (n_cols, n_rows) #originally (7,6) # 4,5 same results
n_rows_and_cols = (n_rows, n_cols)
sqr_size = 0.022 # 22mm

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((n_rows*n_cols,3), np.float32)
objp[:,:2] = np.mgrid[0:n_rows,0:n_cols].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

#mypath = "/home/sampfeiffer/svn/argus_ws/src/argus_tools/data/calibrationdata_camera_1/"
#print "Getting images from " + mypath
#images = glob.glob(mypath + '*.png')
#print "images is: " + str(images)


#ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
with open('calibration.yaml') as f:
    loadeddict = yaml.load(f)
mtx = loadeddict.get('camera_matrix')
print('camera matrix = ',mtx)
dist = loadeddict.get('dist')
print("distortion matrix = ",dist)
#rvecs, tvecs = loadeddict.get(rvecs), loadeddict.get(tvecs)


#cv2.imshow('img',image_data)

while(True):
    #print "\nImage " + fname
#     if idx > 10:
#         break
    _, image_data = camera.read()
    image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(image_data, n_rows_and_cols,None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        print("  found " + str(len(corners)) + " corners.")
        objpoints.append(objp)
        # cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria) didnt work, I couldnt make it work copying the calibrator code
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(image_data, n_rows_and_cols, corners, ret)

        # Find the rotation and translation vectors.
        rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners, mtx, dist)
#         print "rvecs: " + str(rvecs)
#         print "tvecs: " + str(tvecs)
        #print "inliers: " + str(inliers)
        print("6DOF pose using square size of: " + str(sqr_size))
        # Tried tvecs_real = tvecs * sqr_size -> didnt work, very close distances
        #scale = sqr_size * math.sqrt(n_rows*n_rows +  n_cols*n_cols) # does not seem to work, too big distances usually
        #tvecs_real = tvecs * scale
        tvecs_real = tvecs *  sqr_size
        print(tvecs_real)

        cv2.drawChessboardCorners(image_data, n_rows_and_cols, corners, ret)
        axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

        image_data = draw(image_data,corners,imgpts)
#         print "corners -1:"
#         x = corners[0][-1][0]
#         y = corners[0][-1][1] # down right corner is corners[0][0] (i think)
#         print str(x) + " " + str(y)
        bx, by, bz = tvecs_real
        bx = round(bx[0], 3)
        by = round(by[0], 3)
        bz = round(bz[0], 3)
        mystr = "x: " + str(bx) + " y: " + str(by) + " z: " + str(bz)
        print(mystr)
        cv2.putText(image_data,"Chessboard pose (lower right corner)", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0)) # last part is color, this is cyan
        cv2.putText(image_data,mystr, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0))
        roll, pitch, yaw = rvecs
        roll = round(roll[0], 3)
        pitch = round(pitch[0], 3)
        yaw = round(yaw[0], 3)
        oristr = "roll: " + str(roll) + " pitch: " + str(pitch) + " yaw: " + str(yaw)
        cv2.putText(image_data,oristr, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0))
        roll = round(math.degrees(roll), 1)
        pitch = round(math.degrees(pitch), 1)
        yaw = round(math.degrees(yaw), 1)
        oristrdeg = "(deg) roll: " + str(roll) + " pitch: " + str(pitch) + " yaw: " + str(yaw) + ""
        cv2.putText(image_data,oristrdeg, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0))

        cv2.imshow('img',image_data)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

camera.release()
cv2.destroyAllWindows()
