Write a Limelight snapscript python program which detects pollen in an
image.

Pollen is a game element in the FTC BioBuzz game.  They are
essentially yellow pickle balls.

Use the images in pollen_data/images for training.  The directory
pollen_data/labels contains rectangles around the pollen in the
images.  This data in YOLO format.

pollen_data/testing contains some additional images to test your code
on.

Your python program should contain a runPipeline(image, llrobot)
routine which can be run on a Limelight.  Assume llrobot is empty. It
should return largestContour, image, llpython.  In the returned image
put a circle or rectangle around each pollen in the image, preferably
a circle.  You can decide what information to return in llpython.  You
can return an empty largestContour.

You may use numpy and opencv. 

Your python program should also contain a __main__ which reads an
image (filename given on the command line), runs runPipeline, and
displays the modified image.

For each subroutine please add an extensive code comment explaining
what that subroutine does.  Your target audience is a smart high
school student.