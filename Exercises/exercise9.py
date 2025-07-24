"""
    Exercise 9

    In this exercise, we are going to take another approach to
    finding the samples: look for straight edges.

    We are going follow 

      https://www.geeksforgeeks.org/python/line-detection-python-opencv-houghline-method/

    First look up the two OpenCV commands:
     * Canny
     * HoughLinesP
    Don't worry too much about all of parameters.  

"""

import cv2
import numpy as np

filename = "../Pics/pic116.png"

img = cv2.imread(filename)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_yellow = np.array([15, 100,  50])
upper_yellow = np.array([30, 255, 255])
mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

ksize = 31             
borderType = cv2.BORDER_CONSTANT 
mask = cv2.GaussianBlur(mask, (ksize, ksize), borderType) 

threshold = 200
_, mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)

edges = cv2.Canny(mask, 50, 150)

lines = cv2.HoughLinesP(edges, 1, np.pi/180, 20, minLineLength=50, maxLineGap=10)


for line in lines:
    x1, y1, x2, y2 = line[0]
    pt1 = (x1, y1)
    pt2 = (x2, y2)
    cv2.line(img, pt1, pt2, [0, 255, 0], 2)
    
cv2.imshow("edges", edges)
cv2.imshow("original", img)
cv2.imshow("mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows
