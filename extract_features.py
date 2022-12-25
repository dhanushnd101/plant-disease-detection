import os
import numpy as np
import cv2 as cv
from skimage import feature
import pickle

import shutil
import argparse



def segment(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # find the green color 
    #mask_green = cv.inRange(hsv, (36,0,0), (86,255,255))
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


def getfeatures(img):

    segImg, mask = segment(img)

    mean = np.sum(segImg/255, axis=0)
    mean = np.sum(mean, axis=0)/np.sum(mask/255)

    mask3 = np.zeros((mask.shape[0], mask.shape[1], 3))
    mask3[:,:,0] = mask
    mask3[:,:,1] = mask
    mask3[:,:,2] = mask

    std = np.sum(((segImg/255 - mean)*mask3/255)**2, axis=0)
    std = np.sqrt(np.sum(std, axis=0)/np.sum(mask/255))

    skew = np.sum(((segImg/255 - mean)*mask3/255)**3, axis=0)
    skew = np.cbrt(np.sum(skew, axis=0)/np.sum(mask/255))

    kurtosis = np.sum(((segImg/255 - mean)*mask3/255)**4, axis=0)
    kurtosis = (np.sum(kurtosis, axis=0)/np.sum(mask/255))**(1/4)

    # Texture features
    img_grayscale = cv.cvtColor(segImg, cv.COLOR_BGR2GRAY)
    g = feature.greycomatrix(img_grayscale, [5], [0, np.pi/2], levels=256,normed=True, symmetric=True)
    contrast = feature.texture.greycoprops(g,prop='contrast')[0,0]
    dissimilarity = feature.texture.greycoprops(g,prop='dissimilarity')[0,0]
    homogeneity = feature.texture.greycoprops(g,prop='homogeneity')[0,0]
    energy = feature.texture.greycoprops(g,prop='energy')[0,0]
    correlation = feature.texture.greycoprops(g,prop='correlation')[0,0]
    ASM = feature.texture.greycoprops(g,prop='ASM')[0,0]
    return ([mean[0], mean[1], mean[2], std[0], std[1], std[2], skew[0], skew[1], skew[2], kurtosis[0], kurtosis[1], kurtosis[2], contrast, dissimilarity, homogeneity, energy, correlation, ASM])



# img = cv.imread('./archive/plantvillage/PlantVillage/Pepper__bell___Bacterial_spot/0a0dbf1f-1131-496f-b337-169ec6693e6f___NREC_B.Spot 9241.JPG')
# getfeatures(img)

featDict = {}

folders = sorted(os.listdir('./PlantVillage_copy'))
for folder in folders:
    print(folder)
    if(folder != '.DS_Store'):
        featDict[folder] = {}
        img_names = sorted(os.listdir('./PlantVillage_copy/' + folder))
        for img_name in sorted(img_names):
    
            img = cv.imread(os.path.join('./PlantVillage_copy/', folder, img_name))
            try:
                f = getfeatures(img)
                featDict[folder][img_name] = f
            except:
                print("Erroneous image: ", os.path.join('./PlantVillage_copy/', folder, img_name))

with open('featDict_copy.pickle', 'wb') as handle:
    pickle.dump(featDict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    

    
    
    