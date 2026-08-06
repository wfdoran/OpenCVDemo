"""Exercise 11

    In this exercise, you are going to use the Limelight runPipeline
    interface again.  This time, you are going to find artifacts
    from the 2025-2026 season and draw boxes around them.

    Start by looking up the OpenCV commands findContours,
    boundingRect, and rectangle.

    Take another look at the sample code on

    https://docs.limelightvision.io/docs/docs-limelight/pipeline-python/snapscript-pipelines

    on how to apply them.

    Use the first parameter in llrobot to tell the Limelight what color
    artifact to look for.  Use the first parameter of llpython for Limelight
    to tell you if it found one. 

"""

import cv2
import numpy as np

#########################################################################
#  Code on Lime Light
#########################################################################


PURPLE = 0
GREEN = 1

def runPipeline(image, llrobot):
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if llrobot[0] == PURPLE:
        lo = (130, 70, 70)
        hi = (150, 255, 255)
    if llrobot[0] == GREEN:
        lo = (70, 70, 70)
        hi = (90, 255, 255)

    img_thresh = cv2.inRange(img_hsv, lo, hi)

    # On the actual Limelight, you cannot display images only
    # return them.  However, during testing on your PC, you
    # can look at intermediate results if you want to or need
    # to.  Just be sure to comment it out before copying to
    # the Limelight.
    
    # cv2.imshow("thresh", img_thresh)

    contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largestContour = np.array([[]])

    # in llpython, use the first parameter tell you if you found
    # an artifact or not.
    llpython = [0, -1, -1, -1, -1]

    # Find the largest contour and draw a rectangle around it
    if len(contours) > 0:
        largestContour = max(contours, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(largestContour)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 255), 2)
        llpython = [1, x, y, w, h]
        
    
    return largestContour, image, llpython

    
#########################################################################
#  Code on Hub
#########################################################################

files = ["../Pics/pic5.png", "../Pics/pic6.png", "../Pics/pic7.png"]

color2string = {GREEN: "green ", PURPLE: "purple"}

for filename in files:
    for color in [GREEN, PURPLE]:
        img_in = cv2.imread(filename, cv2.IMREAD_COLOR)
        llrobot = [color]
        _, img_out, llpython = runPipeline(img_in, llrobot)
        print(filename, color2string[color], llpython)

        # only display pictures where an artifact of the desired color
        # was found.
        if llpython[0] == 1:
            cv2.imshow(filename + " " + color2string[color], img_out)
            cv2.waitKey(0)

cv2.destroyAllWindows()
