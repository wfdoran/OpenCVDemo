""" Exercise 16

    This is the final exercise in our machine learning with OpenCV
    tour.  In this exercise, we are going to use the SVM to find
    pollen in our test images.  This is often called inference.  

    We are going to use detectMultiScale to do the work.

      

"""

import cv2
import os
import numpy as np


def hog_set_svm(hog, svm):
    """I wish OpenCV has a function

              hog.setSVM(svm)

        but it does not.  You actually have to pull out the support
        vector in the svm and the cut-off threshold.  For now, trust
        that this function does the right thing.
    """
    sv = svm.getSupportVectors()
    sv = np.array(sv, dtype=np.float32)
    rho, alpha, _ = svm.getDecisionFunction(0)
    alpha = np.array(alpha, dtype=np.float32).reshape(-1,1)
    detector = np.matmul(sv.T, alpha) * -1
    detector = np.append(detector.flatten(), rho)

    hog.setSVMDetector(detector)
    

# To decide if a patch from a test image is pollen or not, we have to
# process it like we did with our training images.  The first step is
# converting patchs to feature vectors using HOG.  Start by setting up
# the hog object with the same parameters you used in exercise 14.

# Bonus: Think a better software engineering solution to keep these
# values in sync.
win_size = (???, ???)             # FixME!
BLOCK_SIZE = (???, ???)           # FixME!
BLOCK_STRIDE = (???, ???)         # FixME!
CELL_SIZE = (???, ???)            # FixME!
NBINS = ???                       # FixME!
hog = cv2.HOGDescriptor(win_size, BLOCK_SIZE, BLOCK_STRIDE, CELL_SIZE, NBINS)

# Read in the svm you trained in the previous exercise and tell your
# hog object to use it.
svm = cv2.ml.SVM_load("pollen_model.xml")
hog_set_svm(hog, svm)

# We are going to use hog.detectMultiScale to do the heavy lifting.
# It will scan through the image looking for patches in an image that
# our SVM model says are pollen are record them.
#
# detectMultiScale has a bunch of parameters.  Here are some values I
# got off of a tutorial website.
#
# Bonus: understand what these parameters do and pick better values
# for our pollen-detecting use case.
pollen_class = 0
hit_threshold = 0.0
win_stride = (8, 8) 
padding = (8, 8)
scale = 1.05

test_dir = "../Pics/pollen_data/testing"
for image_file in os.listdir(test_dir):
    # Roll through the images in test directory and
    # read them in.
    full_image_file = os.path.join(test_dir, image_file)
    img = cv2.imread(full_image_file)

    # Have detectMultiScale find pollen.
    rects, weights = hog.detectMultiScale(
        ???,                                        # FixME!
        hitThreshold = hit_threshold,
        winStride = win_stride,
        padding = padding,
        scale = scale)

    # Draw a box around each pollen detected.
    #
    # Bonus: look up what the weights are display them in the picture.
    for r in rects:
        upper_left = (???, ???)                    # FixME!
        lower_right = (???, ???)                   # FixME!
        cv2.rectangle(img, upper_left, lower_right, (0,0,255))
    
    # Display the updated image
    cv2.imshow("image: %s" % (image_file,), img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



    
    


