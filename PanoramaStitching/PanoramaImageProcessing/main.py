#Import Libraries
import cv2
import numpy as np

# Load our images
left = cv2.imread("yellowstone-national-park-left.jpg")
left = cv2.resize(left, (0,0), fx=0.5, fy=0.5)
right = cv2.imread("yellowstone-national-park-right.jpg")
right = cv2.resize(right, (0,0), fx=0.5, fy=0.5)

#To achieve more accurate results we will load our two images as a grayscale.
left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

#Initiative start detector orb fot detect keypoints both images.
orb = cv2.ORB_create(nfeatures=800)

# find the keypoints with orb
kp_left, des_left = orb.detectAndCompute(left_gray,None)
kp_right, des_right = orb.detectAndCompute(right_gray,None)
cv2.imshow('left', cv2.drawKeypoints(left, kp_left, None, (255, 255, 0)))
cv2.imshow('right', cv2.drawKeypoints(right, kp_right , None, (255, 255, 0)))

#create BFMatcher object with default parameters
bf = cv2.BFMatcher_create(cv2.NORM_HAMMING)

#Match descriptors
matches = bf.knnMatch(des_left, des_right, k=2)

# Apply ratio test
good_list = []
good = []
for m,n in matches:
    if m.distance < 0.5*n.distance:
        good_list.append([m])
        good.append(m)

cv2.imshow('matches', cv2.drawMatchesKnn(left_gray, kp_left, right_gray, kp_right, good_list, None, flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS))

minimum_match_count = 10
if len(good) > minimum_match_count:
    # Convert keypoints to an argument for findHomography
    src_pts = np.float32([kp_right[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_left[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)

    # Establish a homography
    H, masked = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

dst = cv2.warpPerspective(right,H,(left.shape[1] + right.shape[1], left.shape[0]))
dst[0:left.shape[0], 0:left.shape[1]] = left
cv2.imshow('final', dst)

gray = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
_,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]
x,y,w,h = cv2.boundingRect(cnt)
crop = dst[y:y+h,x:x+w]

cv2.imshow('final', crop)
cv2.waitKey(0)
cv2.destroyAllWindows()

