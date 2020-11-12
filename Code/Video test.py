import numpy as np
import cv2

params = cv2.SimpleBlobDetector_Params()

# Filter by Area.
params.filterByArea = True
params.minArea = 25
params.maxArea = 5000

params.minDistBetweenBlobs = 3


params.filterByColor = True
params.filterByConvexity = True

# tweak these as you see fit
# Filter by Circularity
# params.filterByCircularity = False
params.minCircularity = 0.75

# # # Filter by Convexity
# params.filterByConvexity = True
# params.minConvexity = 0.87

# Filter by Inertia
params.filterByInertia = True
# params.filterByInertia = False
params.minInertiaRatio = 0.1


# If you decide to use a different checkerboard, change the setting here.
checkerboard_x = 7
checkerboard_y = 6

cap = cv2.VideoCapture(0)
_, image_data = cap.read() #need black and white images
image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY) #This converts the image to black and white
width, height = image_data.shape

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    _, thrash = cv2.threshold(gray, 10, 220, cv2.THRESH_BINARY)
    

    # Detect blobs.
    keypoints = detector.detect(gray)

    im_with_keypoints = cv2.drawKeypoints(gray, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)



    #fig = plt.figure()
    im_with_keypoints = cv2.drawKeypoints(gray, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #plt.imshow(cv2.cvtColor(im_with_keypoints, cv2.COLOR_BGR2RGB),interpolation='bicubic')

    #titlestr = '%s found %d keypoints' % (fname, len(keypoints))
    #plt.title(titlestr)

    #fig.canvas.set_window_title(titlestr)


    ret, corners = cv2.findCirclesGrid(gray, (checkerboard_x, checkerboard_y), flags=(cv2.CALIB_CB_ASYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING ), blobDetector=detector )

    #contours, _ = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    #for contour in contours:
    #    approx = cv2.approxPolyDP(contour, 0.001 * cv2.arcLength(contour, True), True)
    #    cv2.drawContours(gray, [approx], 0, (0, 255, 0),1)

    # Display the resulting frame
    cv2.imshow('frame',gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
