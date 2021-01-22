import math
import time
from collections import deque

import numpy as np
from cv2 import cv2 as cv
from matplotlib import pyplot as plt


MIN_HIST_CORREL = 0.8
MIN_HIST_SIMILAR = 0.5

DOWNSAMPLE = 20
REVERSED = False


color_spaces = (
    ("RGB", cv.COLOR_BGR2RGB),
    ("HSV", cv.COLOR_BGR2HSV),  # H: [0,179[
    ("XYZ", cv.COLOR_BGR2XYZ),
    ("YUV", cv.COLOR_BGR2YUV),  # best results
    ("YCR_CB", cv.COLOR_BGR2YCR_CB),
    ("LUV", cv.COLOR_BGR2LUV),
    ("LAB", cv.COLOR_BGR2LAB),
)

methods = (
    ("CORREL", cv.HISTCMP_CORREL),
    ("BHATTACHARYYA", cv.HISTCMP_BHATTACHARYYA),
    ("CHISQR", cv.HISTCMP_CHISQR),
    ("CHISQR_ALT", cv.HISTCMP_CHISQR_ALT),
    ("HELLINGER", cv.HISTCMP_HELLINGER),
    ("INTERSECT", cv.HISTCMP_INTERSECT),
    ("KL_DIV", cv.HISTCMP_KL_DIV),
)

start = time.time()
cap = cv.VideoCapture('corrupted_video.mp4')
images = []
hists = []

size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))

fps = cap.get(cv.CAP_PROP_FPS)

# get images
while cap.isOpened():
    ret, image = cap.read()
    if not ret:
        break

    images.append(image)

cap.release()


color_space_name, color_space_val = color_spaces[3]
method_name, method_val = methods[0]

print("fps:", fps)
print("color space:", color_space_name)
print("hist compare method:", method_name)
print("number of images:", len(images))

color_images = [cv.cvtColor(image, color_space_val) for image in images]

hists = [cv.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
         for image in color_images]

remove_indices = []

for index in range(len(images)):
    dists = [cv.compareHist(hists[index], hist, method_val)
             for hist in hists]

    dists = sorted(dists, key=lambda distance: distance)

    val = len([dist for dist in dists if dist > MIN_HIST_CORREL])/len(dists)
    # there is less than 50% of images that have a correlation of 0.8 with this image,
    # so we assume that this image is fake
    if val < MIN_HIST_SIMILAR:
        remove_indices.append(index)

print("removed", len(remove_indices), "images")

images = [images[index]
          for index in range(len(images)) if index not in remove_indices]
print("removed", len(remove_indices), "images")

cg_images = [(image, cv.resize(cv.cvtColor(image, cv.COLOR_BGR2GRAY),
                               (size[0]//DOWNSAMPLE, size[1]//DOWNSAMPLE)), index)
             for (index, image) in enumerate(images)]

print((size[0], size[1]), "resized to",
      (size[0]//DOWNSAMPLE, size[1]//DOWNSAMPLE))


def find_min_frame(cg_images: list, old_gray):
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    feature_params = dict(maxCorners=50, qualityLevel=0.05,
                          minDistance=2, blockSize=7)

    old_contours = cv.goodFeaturesToTrack(
        old_gray, mask=None, **feature_params)

    min_dist = 1 << 20
    ret = None
    for index in range(len(cg_images)):
        new_gray = cg_images[index][1]

        new_contours, status, error = cv.calcOpticalFlowPyrLK(
            old_gray, new_gray, old_contours, None, **lk_params)

        good_old = old_contours[status == 1]
        good_new = new_contours[status == 1]

        distance = 0
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            distance += math.sqrt((a-c)**2 + (b-d)**2)

        distance /= len(status)

        if distance < min_dist:
            ret = (index, distance)
            min_dist = distance

    return ret


first_c_g_image = cg_images[0]
del cg_images[0]

ordered = deque()
ordered.append(first_c_g_image)

index, distance = find_min_frame(cg_images, first_c_g_image[1])

ordered.appendleft(cg_images[index])
del cg_images[index]

top_index, top_distance = find_min_frame(cg_images, ordered[+0][1])
bot_index, bot_distance = find_min_frame(cg_images, ordered[-1][1])

while len(cg_images):

    now = time.time()
    message = ""

    if(top_distance < bot_distance):
        ordered.appendleft(cg_images[top_index])
        del cg_images[top_index]

        message += f"top: cg_images {len(cg_images)} left"

        if len(cg_images):
            if bot_index >= top_index and bot_index != 0:
                bot_index -= 1

            top_index, top_distance = find_min_frame(
                cg_images, ordered[+0][1])

    else:
        ordered.append(cg_images[bot_index])
        del cg_images[bot_index]

        message += f"bot: cg_images {len(cg_images)} left"

        if len(cg_images):
            if top_index >= bot_index and top_index != 0:
                top_index -= 1
            bot_index, bot_distance = find_min_frame(cg_images, ordered[-1][1])

    message += f" last: {time.time()-now:.2f}s, total: {time.time()-start:.2f}"

    print(message)

correct_output = []
with open("correct_output.in") as f:
    correct_output = [int(i) for i in f.read().strip().split(",")]

# check if the frames are correctly ordored
print([i[2] for i in ordered] == correct_output)

# for image,gray,id_ in ordered:
#     cv.imshow("adzd", cv.resize(image, (1920//4, 1080//4)))
#     cv.waitKey(0)

if REVERSED:
    ordered.reverse()

filename = "python_video.mp4"
fourcc = cv.VideoWriter_fourcc(*'MJPG')
out = cv.VideoWriter(filename, fourcc, fps, size, True)
print(f"writing {filename} ...")

for color_image, gray_image, id_ in ordered:
    out.write(color_image)

out.release()
cv.destroyAllWindows()

#######
# orb = cv.ORB_create()

# FLANN_INDEX_LSH = 6
# index_params = dict(algorithm=FLANN_INDEX_LSH,
#                     table_number=6,  # 12
#                     key_size=12,     # 20
#                     multi_probe_level=1)  # 2
# search_params = dict(checks=50)

# flann = cv.FlannBasedMatcher(index_params, search_params)

# img1 = images[0]
# del images[0]
# ordered_images = [img1]

# R = []
# for i in range(0, len(images)):
#     img2 = images[i]

#     kp1, des1 = orb.detectAndCompute(img1, None)
#     kp2, des2 = orb.detectAndCompute(img2, None)

#     matches = flann.knnMatch(des1, des2, k=2)

#     # Apply ratio test
#     good = []
#     for match in matches:
#         if len(match) != 2:
#             continue
#         m, n = match
#         if m.distance < 0.75*n.distance:
#             good.append([m])

#     lol = [g[0].distance for g in sorted(good, key=lambda x: x[0].distance)]
#     R.append((img2, lol[0]))
#     print(i)
#     # img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None,
#     #                          flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
#     # plt.imshow(img3)
#     # plt.show()

# R = sorted(R, key=lambda x: x[1])
# print([x[1] for x in R])

# for image, dist in R:
#    plt.imshow(cv.addWeighted(image, 0.7, img1, 0.3, 0))
#    plt.title(dist)
#    plt.show()

########

# src_pts = np.float32(
#     [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
# dst_pts = np.float32(
#     [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
# M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
# x_off, y_off, _ = M[:, 2]
# print("before", x_off)
# if x_off > 0:
#     P.append(img2)

# orb = cv.ORB_create()

# kp1, des1 = orb.detectAndCompute(img1, None)
# kp2, des2 = orb.detectAndCompute(img2, None)

# bf = cv.BFMatcher()
# matches = bf.knnMatch(des1, des2, k=2)

# #matches = sorted(matches, key=lambda x: x.distance)
# #print([x.distance for x in matches])

# good = []
# for m, n in matches:
#     if m.distance < 0.75*n.distance:
#         good.append([m])

# img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2,
#                       good, None, flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
# plt.imshow(img3), plt.show()

# for image, hist in IH:
#     img = image

#     # Initiate ORB detector
#     orb = cv.ORB_create()
#     # find the keypoints with ORB
#     kp = orb.detect(img, None)
#     # compute the descriptors with ORB
#     kp, des = orb.compute(img, kp)
#     # draw only keypoints location,not size and orientation
#     img = cv.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)

#     fig = plt.figure()
#     fig.add_subplot(1, 2, 1)
#     plt.imshow(img)
#     plt.show()

# for image, hist in IH:
#     gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#     corners = cv.goodFeaturesToTrack(gray, 0, 0.01, 10)
#     for corner in corners:
#         x, y = map(int, corner[0])
#         cv.circle(image, (x, y), 3, 255, -1)

#     cv.imshow('dst', image)
#     cv.waitKey(0)

# for idx in range(len(images_hists)-2):

#     image_hist0 = images_hists[idx+0]
#     image_hist1 = images_hists[idx+1]
#     image_hist2 = images_hists[idx+2]

#     d01 = cv.compareHist(image_hist0.hist, image_hist1.hist, method_val)
#     d02 = cv.compareHist(image_hist0.hist, image_hist2.hist, method_val)
#     # d12 = cv.compareHist(image_hist1.hist, image_hist2.hist, method_val)
#     if d02 > d01:
#         images_hists[idx+2], images_hists[idx +
#                                           1] = images_hists[idx+1], images_hists[idx+2]

#     print(d01, d02, )  # d12)

# for image,hist in images_hists:
#     cv.imshow("adzdadz",image)
#     cv.waitKey(0)

# .exit(0)

# ordered_images = deque(images_hists[:2])
# del images_hists[:2]

# base_image, base_hist = images_hists.pop()

# dists = [(cv.compareHist(base_hist, hist, method_val), index)
#          for (index, (image, hist)) in enumerate(images_hists)]

# for image, hist in ordered_images:
#     cv.imshow("adzd", image)
#     cv.waitKey(0)

# cv.destroyAllWindows()

# FEATURE MATCHING
# image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
# ret, thresh = cv.threshold(image, 127, 255, 0)
# blurred = cv.GaussianBlur(image, (3, 3), 0)
# edges = cv.Canny(blurred, 0, 255)

# sigma = 0.33
# v = np.median(blurred)
# lower = int(max(0, (1.0 - sigma) * v))
# upper = int(min(255, (1.0 + sigma) * v))
# automatic_edges = cv.Canny(blurred, lower, upper)

# plt.subplot(221), plt.imshow(image, cmap='gray')
# plt.title('Original Image')
# plt.subplot(222), plt.imshow(thresh, cmap='gray')
# plt.title('Binary Image')
# plt.subplot(223), plt.imshow(edges, cmap='gray')
# plt.title('Edges')
# plt.subplot(224), plt.imshow(automatic_edges, cmap='gray')
# plt.title('Auto Edges')

# plt.show()

# SHOW REMOVE_IMAGES
# fig = plt.figure()
# for i, remove_image in enumerate(remove_images):
#     fig.add_subplot(1, len(remove_images), i+1)
#     plt.imshow(cv.resize(remove_image, (50, 50)))
#     plt.axis("off")
# plt.show()

# print(color_space_name, "median", .median(
#    D), "mean", .mean(D))

# fig = plt.figure()
# fig.suptitle(
#     f"Color Space: {color_space_name}, with compareHist method: {method_name}", fontsize=10)
#
# for i, (image, dist) in enumerate(I):
#     ax = fig.add_subplot(5, len(I)//5+1, (i + 1))
#     ax.set_title(f"{dist:2f}", size=5)
#     plt.imshow(cv.resize(image, (50, 50)))
#     plt.axis("off")


def find_by_face(images_hists):
    face_cascade = cv.CascadeClassifier(
        'haarcascade_frontalface_default.xml')
    lol = []
    for image, hist in images_hists:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        face = max(faces, key=lambda x: x[2]*x[3])

        x, y, w, h = face
        cv.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

        lol.append((image, x+w/2))

    for image, x in sorted(lol, key=lambda x: x[1]):
        cv.imshow('image', cv.resize(image, (1920//4, 1080//4)))
        cv.waitKey(0)


def optical_flow():

    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)
    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    prev_ = cv.cvtColor(images[0], cv.COLOR_BGR2GRAY)

    for i in range(1, len(images)):
        next_ = cv.cvtColor(images[i], cv.COLOR_BGR2GRAY)

        flow = cv.calcOpticalFlowFarneback(
            prev_, next_, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])

        hsv = np.zeros_like(images[0])
        hsv[..., 1] = 255
        hsv[..., 0] = ang*180/np.pi/2
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        cv.imshow('frame2', bgr)

        # # Change here
        # horz = cv.normalize(flow[..., 0], None, 0, 255, cv.NORM_MINMAX)
        # vert = cv.normalize(flow[..., 1], None, 0, 255, cv.NORM_MINMAX)
        # horz = horz.astype('uint8')
        # vert = vert.astype('uint8')

        # cv.imshow('Horizontal Component', horz)
        # cv.imshow('Vertical Component', vert)
        cv.waitKey(0)
