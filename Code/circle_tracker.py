import sys
import os
import cv2
import numpy as np
import yaml
import math

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

def removeShadow(img):
    rgb_planes = cv2.split(img)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)

    return result_norm


#Load Camera Parameters
with open('calibration.yaml') as f:
    loadeddict = yaml.load(f, Loader=yaml.FullLoader)
camera_matrix = np.asarray(loadeddict.get('camera_matrix'))
dist_coeffs = np.asarray(loadeddict.get('dist'))

print("Camera matrix =", camera_matrix)
print('Type 0: ',type(camera_matrix))
print("Distortion Coefficients matrix=", dist_coeffs)

# Original blob coordinates
objp = np.zeros((21, 3), np.float32)
objp[0]  = (0  , 0  , 0)
objp[1]  = (0 , 0.36 , 0)
objp[2]  = (0 , 0.72 , 0)
objp[3]  = (0.18 , 0.18 , 0)
objp[4]  = (0.18 , 0.54 , 0)
objp[5]  = (0.18, 0.9, 0)
objp[6]  = (0.36  , 0  , 0)
objp[7]  = (0.36 , 0.36 , 0)
objp[8]  = (0.36 , 0.72 , 0)
objp[9]  = (0.54 , 0.18 , 0)
objp[10] = (0.54 , 0.54 , 0)
objp[11] = (0.54 , 0.9 , 0)
objp[12] = (0.72 , 0  , 0)
objp[13] = (0.72 , 0.36 , 0)
objp[14] = (0.72 , 0.72 , 0)
objp[15] = (0.9 , 0.18 , 0)
objp[16] = (0.9 , 0.54 , 0)
objp[17] = (0.9 , 0.9 , 0)
objp[18] = (1.08  , 0  , 0)
objp[19] = (1.08 , 0.36 , 0)
objp[20] = (1.08 , 0.72 , 0)

#projected axis
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

########################################Blob Detector##############################################
# Setup SimpleBlobDetector parameters.
blobParams = cv2.SimpleBlobDetector_Params()

# Change thresholds
blobParams.minThreshold = 10
blobParams.maxThreshold = 220

# distance between blobs
blobParams.minDistBetweenBlobs = 2

# Filter by Area.
blobParams.filterByArea = True
blobParams.minArea = 25     # minArea may be adjusted to suit for your experiment
blobParams.maxArea = 5000   # maxArea may be adjusted to suit for your experiment

# Filter by Circularity
blobParams.filterByCircularity = True
blobParams.minCircularity = 0.1

# Filter by Convexity
blobParams.filterByConvexity = True
blobParams.minConvexity = 0.95

# Filter by Inertia
blobParams.filterByInertia = True
blobParams.minInertiaRatio = 0.1

# Filter by COLOR
blobParams.filterByColor = True
blobParams.blobColor = 0
blobParams.minRepeatability = 2

# Create a detector with the parameters
blobDetector = cv2.SimpleBlobDetector_create(blobParams)
###################################################################################################

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# initialize the camera and grab a reference to the raw camera capture
camera = cv2.VideoCapture(0)
_, image_data = camera.read() #need black and white images
image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY) #This converts the image to black and white
width, height = image_data.shape
#cv2.imshow()


while(True):

    _, image_data = camera.read() #need black and white images
    imgRemapped = image_data.copy()
    imgRemapped_gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY) #This converts the image to black and white
    imgRemapped_gray = removeShadow(imgRemapped_gray) # this removes all shadows in image
    th, imgRemapped_gray = cv2.threshold(imgRemapped_gray, 128, 192, cv2.THRESH_OTSU)


    cv2.imwrite('binary_img.png',imgRemapped_gray)

    keypoints = blobDetector.detect(imgRemapped_gray) # Detect blobs.

    # Draw detected blobs as red circles. This helps cv2.findCirclesGrid() .
    im_with_keypoints = cv2.drawKeypoints(imgRemapped, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    im_with_keypoints_gray = cv2.cvtColor(im_with_keypoints, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findCirclesGrid(im_with_keypoints, (3,7), None, flags = cv2.CALIB_CB_ASYMMETRIC_GRID)   # Find the circle grid

    if ret == True:
        corners2 = cv2.cornerSubPix(im_with_keypoints_gray, corners, (11,11), (-1,-1), criteria)    # Refines the corner locations.

        # Draw and display the corners.
        im_with_keypoints = cv2.drawChessboardCorners(imgRemapped, (3,7), corners2, ret)

        # 3D posture
        if len(corners2) == len(objp):
            retval, rvec, tvec = cv2.solvePnP(objp, corners2, camera_matrix, dist_coeffs)

        if retval:
            projectedPoints, jac = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)  # project 3D points to image plane
            projectedAxis, jacAsix = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)    # project axis to image plane
            for p in projectedPoints:
                p = np.int32(p).reshape(-1,2)
                cv2.circle(im_with_keypoints, (p[0][0], p[0][1]), 3, (0,0,255))
            origin = tuple(corners2[0].ravel())
            #im_with_keypoints = draw(im_with_keypoints, corners2, projectedPoints)

            #display pose on image
            bx, by, bz = tvec
            bx = round(bx[0], 3)
            by = round(by[0], 3)
            bz = round(bz[0], 3)
            mystr = "x: " + str(bx) + " y: " + str(by) + " z: " + str(bz)
            print(mystr)
            cv2.putText(im_with_keypoints,"Chessboard pose (lower right corner)", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0)) # last part is color, this is cyan
            cv2.putText(im_with_keypoints,mystr, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
            roll, pitch, yaw = rvec
            roll = round(roll[0], 3)
            pitch = round(pitch[0], 3)
            yaw = round(yaw[0], 3)
            oristr = "roll: " + str(roll) + " pitch: " + str(pitch) + " yaw: " + str(yaw)

            cv2.putText(im_with_keypoints,oristr, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
            roll = round(math.degrees(roll), 1)
            pitch = round(math.degrees(pitch), 1)
            yaw = round(math.degrees(yaw), 1)
            oristrdeg = "(deg) roll: " + str(roll) + " pitch: " + str(pitch) + " yaw: " + str(yaw) + ""
            print(oristrdeg)
            cv2.putText(im_with_keypoints,oristrdeg, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))


            cv2.imshow("circlegrid", im_with_keypoints) # display
            #cv2.waitKey(5000)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        cv2.imshow("Marker not Found, please place marker in Frame", imgRemapped)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.waitKey(2)
