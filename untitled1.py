#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 17:56:07 2021

@author: dhanushdinesh
"""

import os
import numpy as np
import cv2 as cv
from skimage import feature
import pickle

import shutil
import argparse

low_green = np.array([25, 52, 72])
high_green = np.array([102, 255, 255])



def segment(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # find the green color 
    #mask_green = cv.inRange(hsv, low_green, high_green)
    # find the brown color
    mask_brown = cv.inRange(hsv, (8, 60, 20), (30, 255, 200))
    # find the yellow color in the leaf
    mask_yellow = cv.inRange(hsv, (14, 39, 64), (40, 255, 255))

    # find any of the three colors(green or brown or yellow) in the image
    
    #mask = cv.bitwise_or(mask_green, mask_brown)
    mask = cv.bitwise_or(mask_brown, mask_yellow)

    # Bitwise-AND mask and original image

    res = cv.bitwise_and(img,img, mask= mask)
    return res, mask

img = cv.imread('./PlantVillage_copy/Tomato_Bacterial_spot/0b27c03f-b3bc-4d96-9b76-6fbd779404b9___NREC_B.Spot 1799.JPG')
cv.imshow("original Image", img)
segImg,mask = segment(img)
cv.imshow("Masked image", segImg)