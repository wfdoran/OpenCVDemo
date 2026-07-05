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
    cv2.imshow("foo", img_thresh)

    contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largestContour = np.array([[]])
    llpython = [0, -1, -1, -1, -1]

    if len(contours) > 0:
        largestContour = max(contours, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(largestContour)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 255), 2)
        llpython = [1, x, y, w, h]
        
    
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
