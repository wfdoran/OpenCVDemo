"""Exercise 14

    In this exercise, we continue on our path to machine learning
    pollen vs non-pollen.  We are eventually going to feed our data to
    a Support Vector Machine for learning.  Instead of feeding the
    straight image, we are going to feed it a "feature vector".  We
    need some method for converting an image into a vector of
    interesting/meaningful featues.  There are many ways to do this.
    In this exercise, we are going to use HOG (Histogram of Oriented
    Gradients) which is built into OpenCV.

      https://learnopencv.com/histogram-of-oriented-gradients/

      https://www.geeksforgeeks.org/computer-vision/histogram-of-oriented-gradients/
 
    These feature vectors will be 4000+ long.  To visualize the
    fecture vectors, we will project them down to 2 "principle
    components" for display purposes and see that the pollen and
    non-pollen rectangles are different.
   

    Bonus: invent your own feature vector.  Think up some statistics
    that might be relatvent such as "number of yellow pixles."

"""

import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from solution13 import generate_labelled_data

def convert_to_feature_vectors(training_data, data_dir, win_size):
    """ Using HOG, convert the image data into feature vectors.

        training_data     directory path the original images
        data_dir          the labelled data [(image_filename, (class, x1, y1, x2, y2))]
        win_size          normalized size for each image patch

        For each rectangle in data, read in the corresponding image,
        crop out the rectangle, resize to win_size, have hog convert
        to a feature vector.
    """

    # Set up HOG.  
    BLOCK_SIZE = (16, 16)
    BLOCK_STRIDE = (8, 8)
    CELL_SIZE = (8, 8)
    NBINS = 9
    hog = cv2.HOGDescriptor(win_size, BLOCK_SIZE, BLOCK_STRIDE, CELL_SIZE, NBINS)

    # Return arrays
    features = []
    labels = []

    # Roll through the labelled data and get the feature vector
    for image_file, box in training_data:
        images_dir = os.path.join(data_dir, "images")
        full_image_file = os.path.join(images_dir, image_file)
        img = cv2.imread(full_image_file)

        class_id, x1, y1, x2, y2 = box

        crop = img[y1:y2, x1:x2]
        crop = cv2.resize(crop, win_size)

        vector = hog.compute(crop)
        if vector is not None:
            features.append(vector.ravel())
            labels.append(class_id)

    # Convert the return array to numpy arrays.  The next step (train a support
    # vector machine) wants numpy arrays.
    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    return X, y
        

if __name__ == "__main__":
    # Using your code from solution13, read in the labelled data.  
    data_dir = "../Pics/pollen_data"
    win_size = (96, 96)
    neg_per_image = 5

    training_data = generate_labelled_data(data_dir, win_size, neg_per_image)

    # Convert to feature vectors
    X, y = convert_to_feature_vectors(training_data, data_dir, win_size)

    # Use Principal Component Analysis to project the feature vectors down
    # to 2 dimensions.  We won't go into the math.  We will just the
    # implementation in sklearn.
    #
    #  https://en.wikipedia.org/wiki/Principal_component_analysis
    # 
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)

    # Plot the 2D version.  Notice the separation.  The SVM training
    # will notice that separation as well and use it classify
    # pollen/random
    for i in range(len(X2)):
        if y[i] == 0:
            marker = 'o'
            color = 'red'
        else:
            marker = 'x'
            color = 'blue'            
        plt.plot(X2[i][0], X2[i][1], marker=marker, color=color)

    plt.plot([], [], marker = 'o', color = 'red', linestyle='None', label = 'pollen')
    plt.plot([], [], marker = 'x', color = 'blue', linestyle='None', label = 'random')
        
    plt.legend()
    plt.show()


    
    
