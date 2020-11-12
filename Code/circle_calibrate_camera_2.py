import numpy as np
import cv2
import yaml

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

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

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
#objp = np.zeros((4*11,3), np.float32)
#objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

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
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

#images = glob.glob('pict*.jpg')

# initialize the camera and grab a reference to the raw camera capture
camera = cv2.VideoCapture(0)
_, image_data = camera.read() #need black and white images
image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY) #This converts the image to black and white
width, height = image_data.shape
cv2.imwrite('croptest.png',image_data)


# allow the camera to warmup
ret0=[]
j=0
while(True):
    #Get a frame from the camera
    _, image_data = camera.read()
    gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findCirclesGrid(gray, (3,7),None,flags=cv2.CALIB_CB_ASYMMETRIC_GRID, blobDetector=blobDetector)
    # If found, add object points, image points (after refining them)
    if ret == True and np.sum(np.int32(ret0))<15:
        ret0.append(ret)
        print("{} more for proper calibration".format(15-np.sum(np.int32(ret0))))
        objpoints.append(objp.astype('float32'))

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2.reshape(-1, 2).astype('float32'))

        # Draw and display the corners
        img = cv2.drawChessboardCorners(image_data.copy(), (8,8), corners2,ret)
        cv2.imshow('Circle grid found, keep it in frame!',img)
        #cv2.waitKey(1000)
        cv2.imwrite('cal{}.jpg'.format(j),img)
        j+=1
        #rawCapture.truncate(0)
        if(np.sum(np.int32(ret0))==15):
            cv2.waitKey(1)
            break

    else:
        cv2.imshow('Circle grid not found. Press ''q'' to quit.', image_data)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

camera.release()
cv2.destroyAllWindows()

dist = np.array([-0.13615181, 0.53005398, 0, 0, 0]) # no translation
print('Type 0: ',type(dist))
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
distortion_coefficients = np.squeeze(dist)

print('Type 1: ',type(mtx))
print('Type 2: ',type(dist))
# It's very important to transform the matrix to list.
data = {'ret': np.asarray(ret).tolist(),'camera_matrix': np.asarray(mtx).tolist(), 'dist': np.asarray(dist).tolist(), 'rvecs':np.asarray(rvecs).tolist(), 'tvecs':np.asarray(tvecs).tolist()}
with open("calibration.yaml", "w") as f:
    yaml.dump(data, f)


print("Calibration done. Here is what you need in the .ini file:")

print("Camera matrix =",mtx)
print("Distortion Coefficients matrix=", dist)
