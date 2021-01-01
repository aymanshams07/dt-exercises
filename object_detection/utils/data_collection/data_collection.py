#!/usr/bin/env python3

import numpy as np
import cv2
from agent import PurePursuitPolicy
from utils import launch_env, seed
from utils import launch_env, seed, makedirs, display_seg_mask, display_img_seg_mask
import matplotlib.pyplot as plt
from skimage import morphology
import skimage
from PIL import Image, ImageTk
from colorfilters import BGRFilter
#import torch
import sys

DATASET_DIR = "../../dataset"

npz_index = 0
counter = 0
baseheight = 224
width = 224

def save_npz(img, boxes, classes):
    global npz_index
    with makedirs(DATASET_DIR):
        np.savez(f"{DATASET_DIR}/{npz_index}.npz", *(img, boxes, classes))
        npz_index += 1


def clean_segmented_image(seg_img):
    #cv2.imshow(seg_img)

    # # TODO
    # # get b box coordinates for the masks
    # num_images = len(seg_img)
    # boxes = []
    # for i in range(num_images):
    #     pos = np.where(seg_img[i])
    #     xmin = np.min(pos[1])
    #     xmax = np.max(pos[1])
    #     ymin = np.min(pos[0])
    #     ymax = np.max(pos[0])
    #     boxes.append([xmin, ymin, xmax, ymax])

    # Tip: use either of the two display functions found in util.py to ensure that your cleaning produces clean masks
    # (ie masks akin to the ones from PennFudanPed) before extracting the bounding boxes
    pass
    # return boxes


seed(123)
environment = launch_env()

policy = PurePursuitPolicy(environment)

MAX_STEPS = 10
#max_classes = 4

while True:
    obs = environment.reset()
    environment.render(segment=True)
    rewards = []

    nb_of_steps = 0
    #classes = 0

    while True:
        action = policy.predict(np.array(obs))

        obs, rew, done, misc = environment.step(action) # Gives non-segmented obs as numpy array
        output_image = obs

        segmented_obs = environment.render_obs(True)  # Gives segmented obs as numpy array
        resized_seg_obs = cv2.resize(segmented_obs, (224, 224))

        # code start
        kernel = np.ones((5, 5), np.uint8)
        se2 = cv2.morphologyEx(resized_seg_obs, cv2.MORPH_OPEN, kernel)
        #cv2.imshow('Output', se2)
        #cv2.waitKey(0)
        #classes = int(input("Enter classes no: "))

        # color matrix
        list_of_4_colors = [
            ([0,   0, 226], [255, 219, 231]),
            ([226, 111, 100], [255, 185, 252]),
            ([116, 114, 117], [255, 116, 117]),
            ([216, 171,  15], [255, 173, 80])
        ]

        # bounding box matrix
        list_of_bbox = [(48, 135, 527, 165),
                 (87, 152, 434, 198),
                 (0, 101, 619, 371),
                 (238, 76, 639, 251)]

        # Duckies
        red_lower = np.array([0,   0, 226], np.uint8)
        red_upper = np.array([255, 219, 231], np.uint8)
        red_mask = cv2.inRange(se2, red_lower, red_upper)

        # Cones
        green_lower = np.array([226, 111, 100], np.uint8)
        green_upper = np.array([255, 185, 252], np.uint8)
        green_mask = cv2.inRange(se2, green_lower, green_upper)

        # Trucks
        blue_lower = np.array([116, 114, 117], np.uint8)
        blue_upper = np.array([255, 116, 117], np.uint8)
        blue_mask = cv2.inRange(se2, blue_lower, blue_upper)

        # buses
        grey_lower = np.array([216, 171,  15], np.uint8)
        grey_upper = np.array([255, 173,  80], np.uint8)
        grey_mask = cv2.inRange(se2, grey_lower, grey_upper)

        # Morphological Transform, Dilation
        # for each color and bitwise_and operator
        # between imageFrame and mask determines
        # to detect only that particular color
        kernal = np.ones((5, 5), "uint8")

        # For red color
        red_mask = cv2.dilate(red_mask, kernal)
        res_red = cv2.bitwise_and(se2, se2,
                                  mask=red_mask)

        # For green color
        green_mask = cv2.dilate(green_mask, kernal)
        res_green = cv2.bitwise_and(se2, se2,
                                    mask=green_mask)

        # For blue color
        blue_mask = cv2.dilate(blue_mask, kernal)
        res_blue = cv2.bitwise_and(se2, se2,
                                   mask=blue_mask)

        # For grey color
        grey_mask = cv2.dilate(grey_mask, kernal)
        res_grey = cv2.bitwise_and(se2, se2,
                                   mask=grey_mask)

        boxes = []
        labels = []

        # Creating contour to track duckies
        contours, hierarchy = cv2.findContours(red_mask,
                                               cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 300:
                x, y, w, h = cv2.boundingRect(contour)
                mask = red_mask
                # instances are encoded as different colors
                obj_ids = np.unique(mask)
                # first id is the background, so remove it
                obj_ids = obj_ids[1:]

                # split the color-encoded mask into a set
                # of binary masks
                masks = mask == obj_ids[:, None, None]

                # get bounding box coordinates for each mask
                num_objs = len(obj_ids)
                boxes = []
                for i in range(num_objs):
                    pos = np.where(masks[i])
                    xmin = np.min(pos[1])
                    xmax = np.max(pos[1])
                    ymin = np.min(pos[0])
                    ymax = np.max(pos[0])
                    boxes.append([xmin, ymin, xmax, ymax])
                labels.append(1)
                save_npz(se2, boxes, labels)

        # Creating contour to track cones
        contours, hierarchy = cv2.findContours(green_mask,
                                               cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if (area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                mask = green_mask
                # instances are encoded as different colors
                obj_ids = np.unique(mask)
                # first id is the background, so remove it
                obj_ids = obj_ids[1:]

                # split the color-encoded mask into a set
                # of binary masks
                masks = mask == obj_ids[:, None, None]

                # get bounding box coordinates for each mask
                num_objs = len(obj_ids)
                boxes = []
                for i in range(num_objs):
                    pos = np.where(masks[i])
                    xmin = np.min(pos[1])
                    xmax = np.max(pos[1])
                    ymin = np.min(pos[0])
                    ymax = np.max(pos[0])
                    boxes.append([xmin, ymin, xmax, ymax])
                #boxes.append([x, y, w, h])
                labels.append(2)
                save_npz(se2, boxes, labels)

        # Creating contour to track truck
        contours, hierarchy = cv2.findContours(blue_mask,
                                               cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if (area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                mask = blue_mask
                # instances are encoded as different colors
                obj_ids = np.unique(mask)
                # first id is the background, so remove it
                obj_ids = obj_ids[1:]

                # split the color-encoded mask into a set
                # of binary masks
                masks = mask == obj_ids[:, None, None]

                # get bounding box coordinates for each mask
                num_objs = len(obj_ids)
                boxes = []
                for i in range(num_objs):
                    pos = np.where(masks[i])
                    xmin = np.min(pos[1])
                    xmax = np.max(pos[1])
                    ymin = np.min(pos[0])
                    ymax = np.max(pos[0])
                    boxes.append([xmin, ymin, xmax, ymax])
                #boxes.append([x, y, w, h])
                labels.append(3)
                save_npz(se2, boxes, labels)

        # Creating contour to track bus
        contours, hierarchy = cv2.findContours(blue_mask,
                                                cv2.RETR_TREE,
                                                cv2.CHAIN_APPROX_SIMPLE)
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if (area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                mask = blue_mask
                # instances are encoded as different colors
                obj_ids = np.unique(mask)
                # first id is the background, so remove it
                obj_ids = obj_ids[1:]

                # split the color-encoded mask into a set
                # of binary masks
                masks = mask == obj_ids[:, None, None]

                # get bounding box coordinates for each mask
                num_objs = len(obj_ids)
                boxes = []
                for i in range(num_objs):
                    pos = np.where(masks[i])
                    xmin = np.min(pos[1])
                    xmax = np.max(pos[1])
                    ymin = np.min(pos[0])
                    ymax = np.max(pos[0])
                    boxes.append([xmin, ymin, xmax, ymax])
                #boxes.append([x, y, w, h])
                labels.append(4)
                save_npz(se2, boxes, labels)

        #classes = labels
        #save_npz(se2, boxes, classes)
        print('these are the bboxes', boxes)
        print('these are the labels', labels)
        cv2.waitKey(10)
        rewards.append(rew)
        environment.render(segment=int(nb_of_steps / 50) % 2 == 0)

        # TODO boxes, classes = clean_segmented_image(segmented_obs)
        # TODO save_npz(obs, boxes, classes)
        # classes += 1
        nb_of_steps += 1
        if done or nb_of_steps > MAX_STEPS:
            break

# in image for red > blue, blue> green, green <blue