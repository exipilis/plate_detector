import csv
import os
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

import cv2
import numpy as np

MIN_MATCH_COUNT = 9
FLANN_INDEX_KDTREE = 0


def cut(line):
    plate_path = 'platesmania/images/' + line[2].replace('http://', '')
    photo_path = 'platesmania/images/' + line[3].replace('http://', '')
    if not os.path.isfile(plate_path) or not os.path.isfile(photo_path):
        return

    # print(plate_path + ' ' + photo_path)

    sift = cv2.xfeatures2d.SIFT_create()

    img1 = cv2.imread(plate_path)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    img2 = cv2.imread(photo_path)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    kp1, descr1 = sift.detectAndCompute(gray1, None)
    kp2, descr2 = sift.detectAndCompute(gray2, None)

    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(descr1, descr2, k=2)

    # Need to draw only good matches, so create a mask
    matches_mask = [[0, 0] for _ in range(len(matches))]
    good = []
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matches_mask[i] = [1, 0]
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        h, w, _ = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, homography)

        x_min = int(np.min(dst[:, 0, 0]))
        x_max = int(np.max(dst[:, 0, 0]))
        y_min = int(np.min(dst[:, 0, 1]))
        y_max = int(np.max(dst[:, 0, 1]))

        cut = img2[y_min:y_max+1, x_min:x_max+1]

        cv2.imwrite('/tmp/' + line[0] + '.png', cut)


reader = csv.reader(open('platesmania/dataset.csv'), delimiter='\t')

pool = ThreadPool(cpu_count())
pool.map(cut, reader)
