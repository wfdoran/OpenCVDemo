"""Exercise 15

    In this exercise, we use the feature vectors generated in the
    previous exercise to train a Support Vector Machine (SVM) to
    differentiate pollen from non-pollen.  Start by reading
    OpenCV's Introduction to Support Vector Machines

    https://docs.opencv.org/4.4.0/d1/d73/tutorial_introduction_to_svm.html

    Notice how their picture looks a lot like your picture from the
    previous exercise.  That is a good sign!

"""

import cv2

from solution13 import generate_labelled_data
from solution14 import convert_to_feature_vectors

def train_svm(X, y):
    """train_svm trains an Support Vector Machine to differentiate
        two classes based on their feature vectors.

        X      array of feature vectors 
        y      class of each item

        Note: len(X) must equal len(y)

    """

    # Follow the example python code from the above page
    # to set up the svm object.  This is the common pattern
    # in machine learning packages.  You first set up the
    # machine learning tool and set its parameters.  Then
    # you feed it data.
    svm = cv2.ml.SVM_create()               
    svm.setType(cv2.ml.SVM_C_SVC)            
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))

    # One thing, they didn't do was set "C".  As a bonus,
    # look up what this is and play with it.
    # 
    # svm.setC(1.0)

    # Do the training.  Feed it the data.  Bonus: look up what
    # svm.train returns and print a warning if something has gone
    # wrong.
    svm.train(X, cv2.ml.ROW_SAMPLE, y)
    return svm


if __name__ == "__main__":
    # Use your functions from the previous two exercise to get the
    # training data for the SVM.  More accurately, regenerate the
    # training data.
    data_dir = "../Pics/pollen_data"
    win_size = (96, 96)
    neg_per_image = 5

    training_data = generate_labelled_data(data_dir, win_size, neg_per_image)
    X, y = convert_to_feature_vectors(training_data, data_dir, win_size)

    # Do the training
    svm = train_svm(X, y)

    # Let's see how the model does on the training data
    train_pred = svm.predict(X)[1].ravel()

    correct = 0
    wrong = 0
    for pred, actual in zip(train_pred, y):
        if round(pred) == actual:
            correct += 1
        else:
            wrong += 1

    print("correct: %d, wrong: %d" % (correct, wrong))

    # Finally, save the model so we can use it in the next
    # exercise. 
    svm.save("pollen_model.xml")

    
        
