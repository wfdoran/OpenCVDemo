"""Exercise 10

    In this exercise, we are going to search for one image (the
    "template") in another image.  Following exercise 3, the template
    will be a yellow sample with a little bit of boundary.  Then we
    will have matchTemplate look for this in the original image.

    Look up the OpenCV commands matchTemplate and minMaxLoc.

    % python exercise10.py

    Why don't we just just this to find samples?  Because it only
    finds templates in the same orientation.  A sample at 45 degrees
    would not be found.

    Homework (hard): use Exercise 9 to get good rotation points and
      angles.  Perform the rotations.  The OpenCV commands
      getRotationMatrix2D and transform are helpful here.  In each
      rotated image, perform the template search.  If you have
      correctly rotated, the sample should be vertical and
      matchTemplate should find it.

"""

import cv2
import numpy as np

filename = "../Pics/pic1.png"
img = cv2.imread(filename)

# Slice out the template (values from Exercise 3)
template = img[145:318,255:345,:]
cv2.imshow("template", template)

# Search for the template in the image
res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF)

# minMaxLoc returns four things.  We want the highest scoring location
# of the template in the picture.  Figure out which of the outputs
# from minMaxLoc give you the max_val and max_loc.  For the other two
# use _ to indicate that we don't care about this output.
_, max_val, _, max_loc = cv2.minMaxLoc(res)

print("max_val: ", max_val)
print("max_loc: ", max_loc)


# draw a box in the orginal image where the template was found
x = max_loc[0]
y = max_loc[1]
dy = 318 - 145
dx = 345 - 255

box = np.array([(x,y), (x+dx,y), (x+dx,y+dy), (x,y+dy)])
cv2.drawContours(img, [box], 0, (0, 255, 0), 2)

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
