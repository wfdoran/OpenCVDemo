"""

    Exercise 5

    Exercise 4 was a set up for why working with RGB (or BGR) is rarely
    the right way to go when wanting to select pixels of a certain color.
    In stead, it usually pays off to convert your image into HSV

          H  hue              <->  roughly color
          S  saturation       <->  distance from white
          V  value            <->  brightness

    Start by looking up the OpenCV command cvtColor and when parameter
    you need to pass to it convert from BGR to HSV.  Then run the
    program.

    $ python exercise5.py

    Run your cursor around the hsv image and see how well the first
    coordinate tracks with color.  Write down a guess at what range
    of H corresponds to the tdhree colors: blue, red, yellow.  
"""
import cv2

filename = "../Pics/pic1.png"

img = cv2.imread(filename)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.imshow("orig", img)
cv2.imshow("hsv", hsv)
cv2.waitKey(0)
cv2.destroyAllWindows
