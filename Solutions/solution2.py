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

    A: Upper-left corner.

    Q: Is the first coordinate down or over?

    A: The first coordinate is down, the second coordinate is over to
       the right.

    Q: What are the three values for each pixel?  They
       are not RGB, they are ...?

    A: [Blue, Green, Red].  For some reason OpenCV defaults to BGR.

    Exercise: Find a pixel in the blue and yellow samples.

"""

import cv2

filename = "../Pics/pic1.png"
img = cv2.imread(filename)

print("type of img: ", type(img))
print("img.shape: ", img.shape)
print("height: %d, width: %d, depth (bytes): %d" % img.shape)


print("Pixel at (233,75): ", img[233][75])     # A red pixel
print(img[245][540])   # A blue pixel        
print(img[220][320])   # A yellow pixel      

cv2.imshow("image", img)
cv2.waitKey(0)

"""
    Bonus: you can change the image if you want!  Display
    only the red coordinate of each pixel.
"""

n_rows, n_cols, _ = img.shape
for row in range(n_rows):
    for col in range(n_cols):
        b, g, r = img[row][col]
        img[row][col] = [0, 0, r]

cv2.imshow("RED!!!", img)
cv2.waitKey(0)
cv2.destroyAllWindows

