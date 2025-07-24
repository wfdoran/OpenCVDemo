"""Exercise 7

    The image for the previous exercise was pretty clean with good
    light and no random wires in the pictures.  Now lets look at a
    real picture from last season.  Start by running this program
    which pulls out the blue samples

    % python exercise7.py

    It is a bit messy.  We would like to do two things:

    - Remove the small speckles.

    - Fill in the blue sample.

    This is called "Noise Reduction".  OpenCV has a bunch of
    ways to to do this, maybe too many ways.

    Noise reduction usually consists of two parts:

    1) Apply some averaging.  Each pixel is replaced with the average
       of the pixels around it.

    2) Apply some threshold.  Pixels below the threshold go to 0,
       above to 255.

    Following what we did last season, lookup documentation for the
    OpenCV commands GaussianBlur and threshold.  Then play with the
    parameters until you are happy with the result.

    Note: to see the actual averaging matrix used by GaussianBlur, you
    can python compute it for you.  Here is the matrix for ksize=5.

    % python
    >>> import cv2
    >>> gauss_1D = cv2.getGaussianKernel(5, -1)
    >>> gauss_2D = gauss_1D @ gauss_1D.T
    >>> gauss_2D
    >>> gauss_2D
    array([[0.00390625, 0.015625  , 0.0234375 , 0.015625  , 0.00390625],
           [0.015625  , 0.0625    , 0.09375   , 0.0625    , 0.015625  ],
           [0.0234375 , 0.09375   , 0.140625  , 0.09375   , 0.0234375 ],
           [0.015625  , 0.0625    , 0.09375   , 0.0625    , 0.015625  ],
           [0.00390625, 0.015625  , 0.0234375 , 0.015625  , 0.00390625]])
    

    Note: for technical reasons that will arise later, it is
    important that the threshold map values above the threshold
    back to 255.

"""

import cv2
import numpy as np

filename = "../Pics/pic101.png"

img = cv2.imread(filename)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_blue = np.array([100, 100,  50])
upper_blue = np.array([140, 255, 255])
blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

ksize = 31
borderType = cv2.BORDER_CONSTANT
blue_mask = cv2.GaussianBlur(blue_mask, (ksize, ksize), borderType) 

threshold = 200
_, blue_mask = cv2.threshold(blue_mask, threshold, 255, cv2.THRESH_BINARY)


cv2.imshow("orig", img)
cv2.imshow("blue", blue_mask)
cv2.waitKey(0)
cv2.destroyAllWindows

"""
    Bonus 1: Run this program on some of the other pictures from
      last season.  Did over train by only working with one
      picture?
"""

"""
    Bonus 2: go to

      https://docs.opencv.org/3.4/d4/d13/tutorial_py_filtering.html

    and read about other methods in OpenCV to denoise.  Pick one
    and try it.  Maybe it is better than GaussianBlur!
"""
