"""Exercise 5

    Exercise 4 was a set up for why working with RGB (or BGR) is rarely
    the right way to go when wanting to select pixels of a certain color.
    In stead, it usually pays off to convert your image into HSV

          H  hue              <->  roughly color
          S  saturation       <->  distance from white
          V  value            <->  brightness

    Start by looking up the OpenCV command cvtColor and what parameter
    you need to pass to it convert from BGR to HSV.  Then run the
    program.

    $ python exercise5.py

    Run your cursor around the hsv image and see how well the first
    coordinate tracks with color.  In the window, it is the "B" value.
    Write down a guess at what range of H corresponds to the three
    colors: blue, red, yellow.

    Windows users will want to use win_imshow instead of cv2.imshow.

"""
import cv2

filename = "../Pics/pic1.png"
# from gear_up_utils import win_imshow

img = cv2.imread(filename)
hsv = cv2.cvtColor(???, ??????)    # Fix ME!

cv2.imshow("orig", img)
cv2.imshow("hsv", hsv)
#win_show("orig", img)
#win_show("hsv", hsv)

cv2.waitKey(0)
cv2.destroyAllWindows
