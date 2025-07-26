"""Exercise 1

    This is the "hello world" of OpenCV.

    Fill in the ????s to read in an image and display it to a window.
    Search the internet for details on imread and imshow.  Once you
    have set those correctly, test by running

    % python exercise1.py

    Windows users: The linux version of imshow displays coordinate and
    pixel info in the bottom bar.  The windows version does not!
    While not perfect, the utility win_imshow in gear_up_utils will
    print out the coordinate and pixel info to the shell from which
    you launched the program.  Uncomment the "from gear_up_utils..."
    line and use win_imshow() instead of cv2.imread().  They have
    the same interface. 

    Comments:
    1) cv2 is the python library with OpenCV bindings.

    2) waitKey(0) waits until you have focus on the displayed image
       and any key (that what the 0 means) is pressed.

    3) Each pixel correpsonds to three values between 0 and 255
       (inclusive).  Use

         https://rgbcolorpicker.com/

       to confirm the colors match the pixel values.    

    4) Bonus points if you can read it in as a grayscale image.

"""
import cv2
# from gear_up_utils import win_imshow

filename = "../Pics/pic1.png"
label = "Hello World!"

img = cv2.imread(???)     # Fix ME!
cv2.imshow(???, ???)      # Fix ME!
# win_imshow(???, ???)    # Windows: Fix ME!
cv2.waitKey(0)
cv2.destroyAllWindows
