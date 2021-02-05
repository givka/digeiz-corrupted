import math
import time
from collections import deque

import numpy as np
from cv2 import cv2 as cv
from matplotlib import pyplot as plt

INPUT_FILENAME = "./corrupted_video.mp4"
OUTPUT_FILENAME = "./python_order_video.mp4"

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
print(f"reading file {INPUT_FILENAME}")
cap = cv.VideoCapture(INPUT_FILENAME)
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
            if bot_index == top_index:
                bot_index, bot_distance = find_min_frame(
                    cg_images, ordered[-1][1])
            elif bot_index > top_index and bot_index != 0:
                bot_index -= 1

            top_index, top_distance = find_min_frame(
                cg_images, ordered[+0][1])

    else:
        ordered.append(cg_images[bot_index])
        del cg_images[bot_index]

        message += f"bot: cg_images {len(cg_images)} left"

        if len(cg_images):
            if top_index == bot_index:
                top_index, top_distance = find_min_frame(
                    cg_images, ordered[+0][1])
            elif top_index > bot_index and top_index != 0:
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

print(f"writing file {OUTPUT_FILENAME}")
fourcc = cv.VideoWriter_fourcc(*'MJPG')
out = cv.VideoWriter(OUTPUT_FILENAME, fourcc, fps, size, True)

for color_image, gray_image, id_ in ordered:
    out.write(color_image)

out.release()
cv.destroyAllWindows()
