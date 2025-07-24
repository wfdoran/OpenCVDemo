"""Exercise 8

    Following the workflow from last season, after getting the mask
    for the color of interest, we looked for contours in the mask.
    These are the borders around the connected regions in the image.

    Then for each contour, we asked OpenCV for the best (rotated)
    rectangle around that contour.  Best in this case means minimum
    area.

    Before going on, lookup the OpenCV commands
    * findContours
    * drawContours
    * minAreaRect
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

# Find the contours in the mask.  Draw the contours on a copy of mask.
contours, _ = cv2.findContours(???, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Fix ME!
c_img = mask.copy()
cv2.drawContours(c_img, ???, -1, (128), 3)                                       # Fix ME!

# Find the minimum area rectangle and draw it in green in the original
# image.  Also, print out the area rectangle.
for c in contours:
    rect = cv2.minAreaRect(???)                             # Fix ME!
    box = cv2.boxPoints(???)                                # Fix ME!
    box = np.intp(???)                                      # Fix ME!
    cv2.drawContours(img, [???], 0, [0, 255, 0], 2)         # Fix ME!
    print("Area: ", cv2.contourArea(???))                   # Fix ME!
    
cv2.imshow("contours", c_img)
cv2.imshow("rects", img)
cv2.imshow("mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows

"""
    Bonus: open the page

      https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html

    and try one of the other things you can do with a contour.
"""
