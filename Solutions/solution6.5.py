"""Exercise 6.5

   In this exercise, you have the same task as Exercise 6: find the
   sample of the desired color.  However, this time we are going to
   use the interface that the Limelight uses for its snap scripts.

   Start by reading

   https://docs.limelightvision.io/docs/docs-limelight/pipeline-python/snapscript-pipelines

   When running a snap script pipeline, the runPipeline script which
   you write is constantly being run on the Limelight.  It takes two
   inputs: the current image being read by the Limelight and a vector
   of parameters which you pass to it.  In this exercise, that vector
   will contain one value telling the Limelight what color to look for.

   On return, runPipeline sends three things back to the hub: an
   optional contour, an image, and a vector of return values.  In this
   example, we want the returned image to be the mask of the desired
   color.  In the returned vector, have the first value be the number
   of pixels of that color.  We are not going to return a contour in
   this exercise.  We will do that in a later exercise.
"""

import cv2
import numpy as np

#########################################################################
#  Code on Lime Light
#########################################################################

RED = 1
BLUE = 2
YELLOW = 3

def runPipeline(image, llrobot):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # The first paramter tells us what color to look for
    color = llrobot[0]
    if color == RED:
        lower_red1 = np.array([  0, 100,  50])
        upper_red1 = np.array([ 10, 255, 255])
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        lower_red2 = np.array([170, 100,  50])
        upper_red2 = np.array([180, 255, 255])
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = red_mask1 + red_mask2

    if color == YELLOW:
        lower_yellow = np.array([15, 100,  50])
        upper_yellow = np.array([30, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
    if color == BLUE:
        lower_blue = np.array([100, 100,  50])
        upper_blue = np.array([140, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Return an empty contour
    largestContour = np.array([[]])

    # Bonus: follow the example in the Limelight documentation
    # to find the largest contour in the mask and draw box
    # around it. 

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        largestContour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largestContour)
        cv2.rectangle(mask, (x,y), (x+w, y+h), (127), 2)
    

    # set the first value in the returned vector to be
    # how many pixels of that color it found
    llpython = [0,0,0,0,0,0,0,0]
    llpython[0] = np.sum(mask) // 255

    return largestContour, mask, llpython

#########################################################################
#  Code on Hub
#########################################################################

filename = "../Pics/pic1.png"
img = cv2.imread(filename)
cv2.imshow("orig", img)

color2string = {RED: "red", BLUE: "blue", YELLOW: "yellow"}

for color in [RED, BLUE, YELLOW]:
    llrobot = [color, 0, 0, 0, 0, 0, 0, 0]
    _, mask, llpython = runPipeline(img, llrobot)
    cv2.imshow(color2string[color], mask)
    print(color2string[color], llpython[0])

cv2.waitKey(0)
cv2.destroyAllWindows
