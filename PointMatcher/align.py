import cv2 as cv
import numpy as np
import argparse
from math import sqrt

parser = argparse.ArgumentParser(description='Threw ideas at the wall, somehow it all worked.')
parser.add_argument('--input1', help='Path to input image 1.', default='./images/im1.jpg')
parser.add_argument('--input2', help='Path to input image 2.', default='./images/im2.jpg')
parser.add_argument('--homography', help='Path to the homography matrix.', default='./images/H1to3p.xml')
args = parser.parse_args()

img1 = cv.imread(args.input1, cv.IMREAD_COLOR)
img2 = cv.imread(args.input2, cv.IMREAD_COLOR)

if img1 is None or img2 is None:
    print('Could not open or find the images!')
    exit(0)




#sift
sift = cv.xfeatures2d.SIFT_create()

keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

MIN_MATCH_COUNT = 10
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(descriptors_1,descriptors_2,k=2)
nn_match_ratio = 0.9  # Nearest neighbor matching ratio
# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
  if m.distance < nn_match_ratio*n.distance:
    good.append(m)

src_pts = np.float32([ keypoints_1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
dst_pts = np.float32([ keypoints_2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

homography, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 100.0)




detector = cv.ORB_create(10000)
# descriptor = cv.ORB_create()
descriptor = cv.xfeatures2d.BEBLID_create(0.5)
kpts1 = detector.detect(img1, None)
kpts2 = detector.detect(img2, None)
kpts1, desc1 = descriptor.compute(img1, kpts1)
kpts2, desc2 = descriptor.compute(img2, kpts2)

matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE_HAMMING)
nn_matches = matcher.knnMatch(desc1, desc2, 2)
matched1 = []
matched2 = []
for m, n in nn_matches:
    if m.distance < nn_match_ratio * n.distance:
        matched1.append(kpts1[m.queryIdx])
        matched2.append(kpts2[m.trainIdx])
inliers1 = []
inliers2 = []
good_matches = []
inlier_threshold = 50.0 # Distance threshold to identify inliers with homography check
for i, m in enumerate(matched1):
    # Create the homogeneous point
    col = np.ones((3, 1), dtype=np.float64)
    col[0:2, 0] = m.pt
    # Project from image 1 to image 2
    col = np.dot(homography, col)
    col /= col[2, 0]
    # Calculate euclidean distance
    dist = sqrt(pow(col[0, 0] - matched2[i].pt[0], 2) + \
                pow(col[1, 0] - matched2[i].pt[1], 2))
    if dist < inlier_threshold:
        good_matches.append(cv.DMatch(len(inliers1), len(inliers2), 0))
        inliers1.append(matched1[i])
        inliers2.append(matched2[i])
res = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
cv.drawMatches(img1, inliers1, img2, inliers2, good_matches, res)
cv.imwrite("./images/matches.png", res)
inlier_ratio = len(inliers1) / float(len(matched1))
print('Matching Results')
print('*******************************')
print('# Keypoints 1:                        \t', len(kpts1))
print('# Keypoints 2:                        \t', len(kpts2))
print('# Matches:                            \t', len(matched1))
print('# Inliers:                            \t', len(inliers1))
print('# Inliers Ratio:                      \t', inlier_ratio)

height, width, channel = img1.shape
src_pts = np.float32([ inliers1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
dst_pts = np.float32([ inliers2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
homography, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 100.0)
img2_warped = cv.warpPerspective(img1, homography, (width, height))

cv.imwrite("./images/img1_warped.png", img2_warped)