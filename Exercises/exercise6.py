"""Exercise 6

    In HSV, H varies from 0 to 180.  S and V vary from 0 to 255.  For
    each color, there is a fairly narrow range of H which picks out
    that color.  That along with average or better S and V usually
    does a good job of picking out the desired color.

    The OpenCV command inRange picks out all of the pixels whose three
    values (HSV in our case) are within given ranges.  The resulting
    "image" has pixel value of either 255 (in the range) or 0 (not in
    the range).

    I have provided a range for detecting blue.

    % python exercise6.py

    Next, use your values from exercise 5 to pick out yellow
    (or google it).

    Finally, red is a little unfortunate as its range wraps around
    zero and we have to use two calls to inRange and add them toegher
    to get all of the red pixels.  Again use your value (or use
    google).

"""

import cv2
import numpy as np

filename = "../Pics/pic1.png"

img = cv2.imread(filename)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_blue = np.array([100, 100,  50])
upper_blue = np.array([140, 255, 255])
blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

cv2.imshow("orig", img)
cv2.imshow("blue", blue_mask)

# lower_yellow = np.array([???, 100,  50])     # Fix ME!
# upper_yellow = np.array([???, 255, 255])     # Fix ME!
# yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
# cv2.imshow("yellow", yellow_mask)

# lower_red1 = np.array([  0, 100,  50])
# upper_red1 = np.array([???, 255, 255])        # Fix ME!
# red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
# lower_red2 = np.array([???, 100,  50])        # Fix ME!
# upper_red2 = np.array([180, 255, 255])
# red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
# red_mask = red_mask1 + red_mask2
# cv2.imshow("red", red_mask)


cv2.waitKey(0)
cv2.destroyAllWindows

"""

   Bonus: go back to exercise4.py and write a really good
   blue score.

"""

