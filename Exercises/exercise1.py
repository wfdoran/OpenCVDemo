"""
    Exercise 1

    This is the "hello world" of OpenCV.

    Fill in the ????s to read in an image and display it to a window.
    Search the internet for details on imread and imshow.  Once you
    have set those correctly, test by running

    % python exercise1.py

    Comments:
    1) cv2 is the python library with OpenCV bindings.

    2) waitKey(0) waits until you have focus on the displayed image
       and any key (that what the 0 means) is pressed.

    3) Bonus points if you can read it in as a grayscale image.

"""
import cv2

filename = "../Pics/pic1.png"
label = "Hello World!"

img = cv2.imread(?????)     # Fix ME!
cv2.imshow(????, ????)      # Fix ME!
cv2.waitKey(0)
cv2.destroyAllWindows
