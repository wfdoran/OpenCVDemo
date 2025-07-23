"""Exercise 3

    One nice feature of working with numpy arrays is that it is easy
    to pull out slices of a picture (aka a crop).

    The syntax "23:203" pulls out all of values 23 <= x < 203.  You
    can do this for the rows and the columns to get a subimage with
    just those row and columns.

    "45:" means 45 <= x, ":234" means 0 <= x < 234, and ":" means
    everything.

    Start by running the program:

    % python3 exercise3.py

    This displys just the yellow sample,  Work out the ranges
    to pull out the red and blue samples.

"""
import cv2

filename = "../Pics/pic1.png"
img = cv2.imread(filename)

yellow_sample = img[145:318,255:345,:]
cv2.imshow("yellow sample", yellow_sample)

red_sample = img[145:318,40:160,:]
cv2.imshow("red sample", red_sample)
blue_sample = img[145:381,460:575:]
cv2.imshow("blue sample", blue_sample)

""" Bonus: By restructing the third coordinate, we can pull out one
    color similar to the bonus in exercise2.  However, here you get a
    2D array since only one value in the last coordinate is preserved.
    OpenCV views this as a gray scale image.
"""
red_coord = 2
red_only = img[:,:,red_coord]              
print("red_only shape: ", red_only.shape)
cv2.imshow("Red!!!!", red_only)

cv2.waitKey(0)
cv2.destroyAllWindows


