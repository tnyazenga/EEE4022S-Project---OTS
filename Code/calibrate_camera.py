# Simple camera calibrator script. Uses the checkerboard. Pretty much straight implementation of the tutorial here:
# http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html#calibration
# IMPORTANT: USE THE ATTACHED PDF HERE!
import numpy as np
import cv2
import yaml


video_device_number = 0 # Device 0 is the built-in webcam. Adjust it to your equipment if needed.
no_of_corners_to_be_collected = 0 # Collect this many checkerboard positions

# If you decide to use a different checkerboard, change the setting here.
checkerboard_x = 12
checkerboard_y = 8

camera = cv2.VideoCapture(video_device_number)
_, image_data = camera.read() #need black and white images
image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY) #This converts the image to black and white
width, height = image_data.shape


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((checkerboard_x*checkerboard_y,3), np.float32)
objp[:,:2] = np.mgrid[0:checkerboard_y,0:checkerboard_x].T.reshape(-1,2)
print(objp)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# Get images from the webcam and try finding the checkerboard. When done, the camera image disappears.
no_of_collected_corners = 0
print("Looking for the checkerboard...")
while(True):
    #Get a frame from the camera
    _, image_data = camera.read()
    image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)

    #Find checkerboard corners
    success, corners = cv2.findChessboardCorners(image_data, (checkerboard_y,checkerboard_x),None)

    if success:
        no_of_collected_corners += 1
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(image_data, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        image_data = cv2.drawChessboardCorners(image_data, (checkerboard_y,checkerboard_x), corners2, success)
        cv2.imshow('Checkerboard found, keep it in frame!', image_data)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        print ("Collected ", no_of_collected_corners-1, "/", no_of_corners_to_be_collected)
        # Get out of loop when we have enough points.
        if no_of_collected_corners > no_of_corners_to_be_collected:
            print("break")
            break
    else:
        cv2.imshow('Checkerboard not found. Press ''q'' to quit.', image_data)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

camera.release()
cv2.destroyAllWindows()

#Now create what we need
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_data.shape[::-1],None,None)
distortion_coefficients = np.squeeze(dist)

print('Type: ',type(mtx))

# It's very important to transform the matrix to list.
data = {'ret': np.asarray(ret).tolist(),'camera_matrix': np.asarray(mtx).tolist(), 'dist': np.asarray(dist).tolist(), 'rvecs':np.asarray(rvecs).tolist(), 'tvecs':np.asarray(tvecs).tolist()}
with open("calibration.yaml", "w") as f:
    yaml.dump(data, f)


print("Calibration done. Here is what you need in the .ini file:")

print(mtx)
print(dist)
