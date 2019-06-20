#Import the necessary packages
import matplotlib.pyplot as plt
import numpy as np
import imutils
import cv2
#import sys

#Load the image, convert it to grayscale, and blur it slightly
input_img = np.loadtxt("human_corridor_0.txt")

#Convert and write the file in .jpg format
cv2.imwrite('out_img.png', input_img)

#Read the image
image = cv2.imread('out_img.png')

#Apply gaussian blur 
gray = cv2.GaussianBlur(image, (7, 7), 0) 
plt.imshow(gray)
plt.show()
#Find the right threshold to omit everything except human, obstacles and wall
#Invert the image for application and finding contours
ret,thresh1 = cv2.threshold(gray,8,255,cv2.THRESH_BINARY_INV)
plt.imshow(thresh1)
plt.show()

#Crop the ROI for distance measure
imCrop = thresh1[int(91):int(107), int(58):int(130)]
plt.imshow(imCrop)
plt.show()

#These Contours expects 32u or 8u images, so apply this
crop = cv2.cvtColor(imCrop, cv2.COLOR_RGB2GRAY)
plt.imshow(crop)
plt.show()
cnts = cv2.findContours(crop.copy(), 1, 2)
# RETR_LIST(1) = retrieves all of the contours without establishing any hierarchical relationships.
# CHAIN_APPROX_SIMPLE(2) = compresses horizontal, vertical, and diagonal segments and leaves only their end points. For example, an up-right rectangular contour is encoded with 4 points.

#Extract the contours values for distance measure
cnts = imutils.grab_contours(cnts)

#Distance measure formula
def minDistance(contour, contourOther):
    #Exchange 1000 to any value depending on the max size of the image
    distanceMin = 1000
    for xA, yA in contour[0]:
        for xB, yB in contourOther[0]:
            distance = ((xB-xA)**2+(yB-yA)**2)**(1/2) # distance formula
            if (distance < distanceMin):
                distanceMin = distance
    return distanceMin

right_dist = minDistance(contour=cnts[0], contourOther=cnts[1])*1.5/60 #distance between human and wall
left_dist = minDistance(contour=cnts[2], contourOther=cnts[1])*1.5/60 #distance between human and shelves

if left_dist > right_dist:
    print("left {}".format(left_dist))
else:
    print("right {}".format(right_dist))
