#  https://docs.limelightvision.io/docs/docs-limelight/pipeline-python/snapscript-pipelines

import cv2
import numpy as np

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
    img_blur = cv2.GaussianBlur(img_thresh, (7, 7), 1.5)

    circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                               param1=100, param2=30, minRadius=0, maxRadius=10000)

    largestContour = np.array([[]])
    llpython = [0, -1, -1, -1, -1]

    if circles is not None:
        for x, y, r in circles[0]:
            x = int(x)
            y = int(y)
            r = int(r)
            cv2.circle(image, (x, y), r, (0, 255, 0), 2)
            cv2.circle(image, (x, y), 2, (0, 0, 255), 3)
            if r > llpython[3]:
                llpython = [1, x, y, r, -1]

    return largestContour, image, llpython

    


files = ["../Pics/pic5.png", "../Pics/pic6.png", "../Pics/pic7.png"]


for filename in files:
    for color in [GREEN, PURPLE]:
        img_in = cv2.imread(filename, cv2.IMREAD_COLOR)
        llrobot = [color]
        _, img_out, llpython = runPipeline(img_in, llrobot)
        print(filename, color, llpython)
        cv2.imshow(filename+str(color), img_out)
        cv2.waitKey(0)
