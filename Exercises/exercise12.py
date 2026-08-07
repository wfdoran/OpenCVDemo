"""Exercise 12

    In this exercise, we are going to learn about the YOLO format for
    labeled images.  The file structure for the YOLO samples is
    simple.  There is an images subdirectory containing the pictures.
    Typically .jpg, .jpeg, and .png are supported.  Then in the labels
    directory there is a corresponding .txt file for each image with
    the same stem name.  Each line in a labels file represents one
    marked rectangle on the image.  Each line has 5 values.  The first
    is what class this subimage is in.  In this example, there is only
    one class: class 0 "pollen".  The next two values are x and y of
    the center of the rectangle, but they are normalized to be between
    0 and 1.  The last two values are the width and height of the
    rectangle, again normalized to be between 0 and 1.

        IMG_4001.txt
          0 0.718067 0.183455 0.062955 0.085963
          0 0.280795 0.588491 0.103155 0.141586

       /data_dir
           classes.txt
           /images
               image1.jpg
               image2.jpg
                ...
           /labels
               label1.txt
               label2.txt

    Finally, there is a classes.txt file which contains the name of
    each class.  Some more details about YOLO are available at

       https://roboflow.com/formats/yolo


    I generated the labels for the images using the utility
    anylabeling (https://github.com/vietanhdev/anylabeling).  This
    tool has a GUI interface which allowed me to click and drag a
    rectangle around each of the pollen in the images and save that
    information in the YOLO format.
   
    In this exercise, your goal is to see how well or poorly I did at
    drawing rectangles.  You will read in each image, read in its
    labels file, and draw the rectangles on the image.  The tricky
    bit is converting the 0-to-1 ratios into the upper-left and
    lower-right in pixels that OpenCV's rectangle command wants.

"""

import os
import cv2

# Set up the paths to the images and labels
data_dir = "../Pics/pollen_data"
images_dir = os.path.join(data_dir, "images")
labels_dir = os.path.join(data_dir, "labels")

# Eventually, you will draw a 2 pixel wide blue rectangle around each
# pollen.
blue = (???,???,???)    # Fix ME!
line_width = ???        # Fix ME!

# loop over all of the images. 
for image_file in os.listdir(images_dir):
    # Set up the full path to the image file and the label file.
    stem = image_file.split('.')[0]
    label_file = stem + ".txt"

    full_image_file = os.path.join(images_dir, image_file)
    full_label_file = os.path.join(???, ???)                # Fix ME!

    # Read in the image and get its height and width.  We will use
    # these value to scale the YOLO values.
    img = cv2.imread(full_image_file)
    img_height = img.shape[???]              # Fix ME!
    img_width = img.shape[???]               # Fix ME!

    # Read in the labels, one line at a time.
    with open(full_label_file) as f:
        for line in f:
            fields = line.split()
            if len(fields) < 5:
                continue

            # The first field is the class_id
            class_id = int(fields[???])          # Fix ME!

            # The next four values are x/y center, width, height.
            # Remember these are ratios from 0 to 1.
            x_center, y_center, w, h = (float(a) for a in fields[???:???])  # Fix ME!

            # Compute the upper-left corner and lower-right corner in
            # pixels.  Convert the ratios into pixel values
            x1 = (x_center - w / 2.0) * img_width
            y1 = (y_center ??? h / 2.0) * img_height   # Fix ME!
            x2 = (x_center + w / 2.0) * ???            # Fix ME!
            y2 = (??? + ??? / 2.0) * ???               # Fix ME!

            # normalize: round to integers and make sure the values
            # are between [0, img_width) and [0, img_height).
            x1 = max(0, int(round(x1)))
            y1 = max(0, int(round(y1)))
            x2 = min(img_width - 1, int(round(x2)))
            y2 = min(img_height - 1, int(round(y2)))

            # a sanity check
            assert(x1 <= x2 and y1 <= y2)

            # draw a rectangle
            upper_left = (???, ???)            # Fix ME!
            lower_right = (???, ???)           # Fix ME!
            cv2.rectangle(img, upper_left, lower_right, blue, line_width)

    # display the image with a rectangle around each pollen
    cv2.imshow("image: %s" % (image_file,), img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

    
