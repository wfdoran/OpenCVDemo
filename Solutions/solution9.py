"""Exercise 9

    In this exercise, we are going to take another approach at finding
    the samples: look for straight lines.  Hopefully, they correspond
    to the edges of samples. 

    We are going follow the tutorial

      https://www.geeksforgeeks.org/python/line-detection-python-opencv-houghline-method/

    First look up the two OpenCV commands:

     * Canny

     * HoughLinesP

    Don't worry too much about all of parameters, we will just follow
    the values in above write up.

    First set the parameters for canny_thresh1 and canny_thresh2 and
    see what the edges image looks like.  I would suggest using the
    values from that geeksforgeeks tutorial.  You can read

      https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html

    for a little more detail on Canny.  Run this to see what Canny()
    produces.

      % python exercise9.py

    Next, uncomment and choose parameters for the HoughLinesP.  For
    our pictures, I think you can do better than values in the
    geeksforgeeks tutorial.  Play around.

    Write each line back on the original image and see how well they
    do.  Also, print out the length of each line.  Look up in python
    math library how to do this

    Some more info on the math behind the the Hough Line Transform. 
     
       https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html

    Bonus: also print out the center point and angle to each line.

"""

import cv2
import numpy as np
import math

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

canny_thresh1 = 50
canny_thresh2 = 150
edges = cv2.Canny(mask, canny_thresh1, canny_thresh2)

hough_thresh = 20
min_line_len = 50
max_line_gap = 10
lines = cv2.HoughLinesP(edges, 1, np.pi/180, hough_thresh, minLineLength=min_line_len, maxLineGap=max_line_gap)

for line in lines:
    x1, y1, x2, y2 = line[0]
    pt1 = (x1, y1)
    pt2 = (x2, y2)
    color = [255, 0, 255]   # purple
    thickness = 2
    cv2.line(img, pt1, pt2, color, thickness)
    line_len = math.dist(pt1,pt2)
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    if angle < -90:
        angle += 180
    if angle > 90:
        angle -= 180
    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    print(line, "%.2f" % (line_len,), center, angle)
    
    
cv2.imshow("edges", edges)
cv2.imshow("original", img)
cv2.imshow("mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows
