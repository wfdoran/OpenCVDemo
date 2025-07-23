"""Exercise 4

    A common theme in FTC is detecting the location of some object at
    the start of auto.  Look at pic2.png, pic3.png, and pic4.png in
    the Pics directory.  Your job is to determine if the blue sample
    is on the left, right, or in the center.

    One more cool feature of numpy arrays is that you sum all of the
    entries.  numpy allows you to perform very fast group operations.

"""
import cv2
import numpy as np

def bad_score_blue(sub_img):
    blue_coord = 0
    return np.sum(sub_img[:,:,blue_coord])

def good_score_blue(sub_img):
    blue_coord = 0
    red_coord = 2
    return int(np.sum(sub_img[:,:,blue_coord])) - int(np.sum(sub_img[:,:,red_coord]))

def score_blue(sub_img):
    return good_score_blue(sub_img)

def blue_location(img):
    _, n_cols, _ = img.shape

    # Compute the cuts for the three regions
    cut1 = n_cols // 3
    cut2 = 2 * n_cols // 3

    # Score each region based on how blue there is
    score_left   = score_blue(img[:,:cut1,:])   
    score_center = score_blue(img[:,cut1:cut2,:])
    score_right  = score_blue(img[:,cut2:,:])

    # return the region with the highest score
    if score_left >= score_center and score_left > score_right:
        return "left"
    if score_right >= score_center:
        return "right"
    if True:
        return "center"

    return "BUG: this should not happen!"

path = "../Pics/"
for filename in ["pic2.png", "pic3.png", "pic4.png"]:
    img = cv2.imread(path + filename)
    print(filename, " is ", blue_location(img))

        
"""

    Bonus: Well... that didn't work very well.  It only got one of
    three correct.  The problem is that the gray in tiles also has a
    lot of blue.  Can you construct a better score_blue function?  Use
    exercise1.py to open one of the files and look at the pixels.
    Notice red level in the tiles versus the blue sample.

"""
