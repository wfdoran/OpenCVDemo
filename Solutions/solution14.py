import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from solution13 import generate_labelled_data

def convert_to_feature_vectors(trainging_data, data, win_size):
    BLOCK_SIZE = (16, 16)
    BLOCK_STRIDE = (8, 8)
    CELL_SIZE = (8, 8)
    NBINS = 9
    hog = cv2.HOGDescriptor(win_size, BLOCK_SIZE, BLOCK_STRIDE, CELL_SIZE, NBINS)
    
    features = []
    labels = []

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

    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)

    return X, y
        

if __name__ == "__main__":
    data_dir = "../Pics/pollen_data"
    win_size = (96, 96)
    neg_per_image = 5

    training_data = generate_labelled_data(data_dir, win_size, neg_per_image)

    X, y = convert_to_feature_vectors(training_data, data_dir, win_size)

    
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)


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


    
    
