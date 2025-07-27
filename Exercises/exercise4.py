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

def score_blue(sub_img):
    blue_coord = ???                  # Fix ME!
    return np.sum(sub_img[:,:,blue_coord])

def blue_location(img):
    _, n_cols, _ = img.???            # Fix ME!

    # Compute the cuts for the three regions
    cut1 = n_cols // 3
    cut2 = ????                       # Fix ME!

    # Score each region based on how blue there is
    score_left   = score_blue(img[:,:cut1,:])   
    score_center = score_blue(img[:,???:???,:])     # Fix ME!
    score_right  = score_blue(img[:,???:???,:])     # Fix ME!
    print(score_left, score_center, score_right)

    # return the region with the highest score
    if ????:                          # Fix ME!
        return "left"
    if ????:                          # Fix ME!
        return "right"
    if ????:                          # Fix ME!
        return "center"

    return "BUG: this should not happen!"

path = "../Pics/"
files = ["pic2.png", "pic3.png", "pic4.png"]
correct = ["right", "center", "left"]
for i in range(3):
    img = cv2.imread(path + files[i])
    print(files[i], " is ", blue_location(img), "correct: ", correct[i])

        
"""

    Bonus: Well... that didn't work very well.  It only got one of
    three correct.  The problem is that the gray in tiles also has a
    lot of blue.  Can you construct a better score_blue function?  Use
    exercise1.py to open one of the files and look at the pixels.
    Notice red level in the tiles versus the blue sample.

"""
