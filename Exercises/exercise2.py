"""Exercise 2

    Color images in OpenCV are just 3-dim numpy arrays!

    First run the program:

    % python exercise2.py

    Now play play around.  For linux users, when you mouse over a
    pixel, information about that pixel will show up in bottom bar of
    the windows.  For windows users, make sure to use win_imshow.
    When you click on pixel, infom about that pixel will printed
    to the shell. 

    Q: What corner is img[0][0]?

    Q: Is the first coordinate down or over?

    Q: What are the three values for each pixel?  They
       are not RGB, they are ...?

    Exercise: Find a pixel in the blue and yellow samples.

"""

import cv2
# from gear_up_utils import win_imshow

filename = "../Pics/pic1.png"
img = cv2.imread(filename)

print("type of img: ", type(img))
print("img.shape: ", img.shape)
print("height: %d, width: %d, depth (bytes): %d" % img.shape)


print("Pixel at (233,75): ", img[233][75])     # A red pixel
# print(img[???][???])   # A blue pixel         Fix ME!
# print(img[???][???])   # A yellow pixel       Fix ME!

cv2.imshow("image", img)
# win_imshow("image", img)
cv2.waitKey(0)

"""
    Bonus: you can change the image if you want!  Display
    only the red coordinate of each pixel.
"""

# n_rows, n_cols, _ = img.shape
# for row in range(n_rows):
#     for col in range(n_cols):
#         b, g, r = img[row][col]
#         img[row][col] = [???, ???, ???]        Fix ME!
#    
# cv2.imshow("RED!!!", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

