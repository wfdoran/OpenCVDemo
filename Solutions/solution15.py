import cv2

from solution13 import generate_labelled_data
from solution14 import convert_to_feature_vectors

def train_svm(X, y):
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setC(1.0)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
    svm.train(X, cv2.ml.ROW_SAMPLE, y)
    return svm


if __name__ == "__main__":
    data_dir = "../Pics/pollen_data"
    win_size = (96, 96)
    neg_per_image = 5

    training_data = generate_labelled_data(data_dir, win_size, neg_per_image)
    X, y = convert_to_feature_vectors(training_data, data_dir, win_size)

    svm = train_svm(X, y)
    
    train_pred = svm.predict(X)[1].ravel()

    correct = 0
    wrong = 0
    for pred, actual in zip(train_pred, y):
        if round(pred) == actual:
            correct += 1
        else:
            wrong += 1

    print("correct: %d, wrong: %d" % (correct, wrong))

    svm.save("pollen_model.xml")

    
        
