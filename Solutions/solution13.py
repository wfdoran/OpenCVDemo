import os
import cv2
import random


def rectangles_overlap(rect_a, rect_b):
    ax1, ay1, ax2, ay2 = rect_a
    bx1, by1, bx2, by2 = rect_b

    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1

def any_overlap(boxes, r_box):
    for box in boxes:
        if rectangles_overlap(box, r_box):
            return True
    return False

def generate_labelled_data(data_dir, win_size, neg_per_image):
    # Set up the paths to the images and labels
    images_dir = os.path.join(data_dir, "images")
    labels_dir = os.path.join(data_dir, "labels")


    training_data = []
    for image_file in os.listdir(images_dir):
        # Set up the full path to the image file and the label file.
        stem = image_file.split('.')[0]
        label_file = stem + ".txt"

        full_image_file = os.path.join(images_dir, image_file)
        full_label_file = os.path.join(labels_dir, label_file)

        # Read in the image and get its height and width.  We will use
        # these value to scale the YOLO values.
        img = cv2.imread(full_image_file)
        img_height = img.shape[0]
        img_width = img.shape[1]
    
        with open(full_label_file) as f:
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
                x2 = min(img_width - 1, int(round(x2)))
                y2 = min(img_height - 1, int(round(y2)))

                training_data.append((image_file, (class_id, x1, y1, x2, y2)))
                pollen.append((x1, y1, x2, y2))

            for _ in range(neg_per_image):
                x1 = random.randint(0, img_width - win_size[0] - 1)
                y1 = random.randint(0, img_height - win_size[1] - 1)
                x2 = x1 + win_size[0]
                y2 = y1 + win_size[1]

                random_box = (x1, y1, x2, y2)
                if not any_overlap(pollen, random_box):
                    training_data.append((image_file, (-1, x1, y1, x2, y2)))

    return training_data

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
     

