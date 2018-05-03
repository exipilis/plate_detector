from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

import cv2
import numpy as np

MIN_MATCH_COUNT = 9
FLANN_INDEX_KDTREE = 0


def match_two(descr1, descr2):
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    try:
        matches = flann.knnMatch(descr1, descr2, k=2)
        k = 0
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                k += 1

        return k

    except:
        print('opencv error')
        return 0


def match_list(sift, gray_img, gray_images):
    _, descr_img = sift.detectAndCompute(gray_img, None)
    descr_images = [sift.detectAndCompute(g, None)[1] for g in gray_images]
    match_counts = np.array([match_two(descr_img, d) for d in descr_images])
    if not len(match_counts):
        return -1

    best_match_id = np.argmax(match_counts)
    if match_counts[best_match_id] < 10:
        best_match_id = -1

    return best_match_id


def process(line):
    sift = cv2.xfeatures2d.SIFT_create()

    line = line.strip().split(',')
    plate_fn = line[1]
    pieces_files = line[2:]

    try:
        plate_img = cv2.imread(plate_fn)
        plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    except:
        print('cv2 except: ' + plate_fn)

    images = [cv2.imread(f) for f in pieces_files]
    gray_images = [cv2.cvtColor(imf, cv2.COLOR_BGR2GRAY) for imf in images]

    best_match_id = match_list(sift, plate_gray, gray_images)

    # print(best_match_id)
    if best_match_id >= 0:
        print(plate_fn + ',' + pieces_files[best_match_id])


filenames = open('plates.csv').readlines()

pool = ThreadPool(cpu_count())
pool.map(process, filenames)

# def cut(line):
#     plate_path = 'platesmania/images/' + line[2].replace('http://', '')
#     photo_path = 'platesmania/images/' + line[3].replace('http://', '')
#     if not os.path.isfile(plate_path) or not os.path.isfile(photo_path):
#         return
#
#     # print(plate_path + ' ' + photo_path)
#
#     sift = cv2.xfeatures2d.SIFT_create()
#
#     img1 = cv2.imread(plate_path)
#     gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#
#     img2 = cv2.imread(photo_path)
#     gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#
#     kp1, descr1 = sift.detectAndCompute(gray1, None)
#     kp2, descr2 = sift.detectAndCompute(gray2, None)
#
#     index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
#     search_params = dict(checks=50)  # or pass empty dictionary
#
#     flann = cv2.FlannBasedMatcher(index_params, search_params)
#
#     matches = flann.knnMatch(descr1, descr2, k=2)
#
#     # Need to draw only good matches, so create a mask
#     matches_mask = [[0, 0] for _ in range(len(matches))]
#     good = []
#     # ratio test as per Lowe's paper
#     for i, (m, n) in enumerate(matches):
#         if m.distance < 0.7 * n.distance:
#             matches_mask[i] = [1, 0]
#             good.append(m)
#
#     if len(good) > MIN_MATCH_COUNT:
#         src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
#         dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
