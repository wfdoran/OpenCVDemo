"""Exercise 13

    In this exercise, we are going to create the training data on
    which to apply some machine learning technique.  The YOLO format
    gives use patches in the various pictures which contain pollen.
    We also need negative sample.  To obtain those, we are going to
    pick 5 random 96 pixel by 96 pixel regions in each picture and
    check whether they overlaps any of the regions marked as
    containing pollen.  If a given patch does not overlap any mark
    pollen region, we will record it as a negative.

    
"""

import os
import cv2
import random


def rectangles_overlap(rect_a, rect_b):
    """determines if two rectangles overlap.

        The format of the rectangles records the coordinates of the
        upper-left and lower-right corner.
    
        (upper_left_x, upper_left_y, lower_right_x, lower_right_y)

        One way they might not over lap is A is to the left of the B

             A1 --+
             |    |
             |    |         B1 --+
             +--- A2        |    |
                            |    |
                            +---B2
        
        So, if A2x <= B1x, the rectangles do not overlap.  There are
        three other configurations would indicate that rectangles do
        not overlap.  In order to overlap, none of these can hold.

        Return TRUE is the rectangles overlap.
    """

    A1x, A1y, A2x, A2y = rect_a
    B1x, B1y, B2x, B2y = rect_b

    return A1x < B2x and A2x > B1x and A1y < B2y and A2y > B1y
    
def any_overlap(boxes, r_box):
    """ determine r_box overlaps with any of the recrangles in box.
    """
    for box in boxes:
        if rectangles_overlap(box, r_box):
            return True
    return False

def generate_labelled_data(data_dir, win_size, neg_per_image):
    """data_dir        directory of the YOLO data
        win_size        size of the random patches
        neg_per_image   how many random samples to try
                        in each image

        Return array
         [(image_filename, (class, x1, y1, x2, y2))]

        The class is 0 for pollen or -1 for random.
        (x1, y1, x2, y2) corners of the rectangle.

        We follow exercise12 to get the positive pollen regions.  Then
        we will generate neg_per_image random regions.  If they do not
        overlap any of the pollen regions, add them.

    """
    # Set up the paths to the images and labels
    images_dir = os.path.join(data_dir, "images")
    labels_dir = os.path.join(data_dir, "labels")

    # training_data[] holds the returned data 
    training_data = []
    for image_file in os.listdir(images_dir):
        # Set up the full path to the image file and the label file.
        stem = os.path.splitext(image_file)[0]
        label_file = stem + ".txt"

        full_image_file = os.path.join(images_dir, image_file)
        full_label_file = os.path.join(labels_dir, label_file)

        # Read in the image and get its height and width.  We will use
        # these value to scale the YOLO values.
        img = cv2.imread(full_image_file)
        img_height = img.shape[0]
        img_width = img.shape[1]
    
        with open(full_label_file) as f:
            # pollen will contain the pollen rectangles in this image.
            pollen = []
            for line in f:
                fields = line.split()
                if len(fields) < 5:
                    continue

                # The first field is the class_id
                class_id = int(fields[0])

                # The next four values are x/y center, width, height.
                # Remember these are ratios from 0 to 1.
                x_center, y_center, w, h = (float(a) for a in fields[1:5])

                # Compute the upper-left corner and lower-right corner in
                # pixels.  Convert the ratios into pixel values
                x1 = (x_center - w / 2.0) * img_width
                y1 = (y_center - h / 2.0) * img_height
                x2 = (x_center + w / 2.0) * img_width
                y2 = (y_center + h / 2.0) * img_height

                # normalize: round to integers and make sure the values
                # are between [0, img_width) and [0, img_height).
                x1 = max(0, int(round(x1)))
                y1 = max(0, int(round(y1)))
                x2 = min(img_width, int(round(x2)))
                y2 = min(img_height, int(round(y2)))

                training_data.append((image_file, (class_id, x1, y1, x2, y2)))
                pollen.append((x1, y1, x2, y2))

            for _ in range(neg_per_image):
                # Pick a random rectangle
                x1 = random.randint(0, img_width - win_size[0] - 1)
                y1 = random.randint(0, img_height - win_size[1] - 1)
                x2 = x1 + win_size[0]
                y2 = y1 + win_size[1]

                # If it does not overlap any pollen, record it with
                # class -1
                random_box = (x1, y1, x2, y2)
                if not any_overlap(pollen, random_box):
                    training_data.append((image_file, (-1, x1, y1, x2, y2)))

    return training_data

# By having a __main__, we can import solution13.py and use
# generate_labelled_data in a later program.
if __name__ == "__main__":
    data_dir = "../Pics/pollen_data"
    win_size = (96, 96)
    neg_per_image = 5

    training_data = generate_labelled_data(data_dir, win_size, neg_per_image)
    
    for image_file, box in training_data:
        images_dir = os.path.join(data_dir, "images")
        full_image_file = os.path.join(images_dir, image_file)
        img = cv2.imread(full_image_file)

        class_id, x1, y1, x2, y2 = box

        crop = img[y1:y2, x1:x2]
        cv2.imshow("class: %d" % (class_id,), crop)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
     

