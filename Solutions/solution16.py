import cv2
import os
import numpy as np

win_size = (96, 96)
BLOCK_SIZE = (16, 16)
BLOCK_STRIDE = (8, 8)
CELL_SIZE = (8, 8)
NBINS = 9
hog = cv2.HOGDescriptor(win_size, BLOCK_SIZE, BLOCK_STRIDE, CELL_SIZE, NBINS)

svm = cv2.ml.SVM_load("pollen_model.xml")
sv = svm.getSupportVectors()
sv = np.array(sv, dtype=np.float32)


rho, alpha, _ = svm.getDecisionFunction(0)
alpha = np.array(alpha, dtype=np.float32).reshape(-1,1)
detector = np.matmul(sv.T, alpha) * -1
detector = np.append(detector.flatten(), rho)

hog.setSVMDetector(detector)
print(svm)


pollen_class = 0
hit_threshold = 0.0
win_stride = (8, 8) 
padding = (8, 8)
scale = 1.05

test_dir = "../Pics/pollen_data/testing"
for image_file in os.listdir(test_dir):
    full_image_file = os.path.join(test_dir, image_file)
    img = cv2.imread(full_image_file)
    rects, weights = hog.detectMultiScale(
        img,
        hitThreshold = hit_threshold,
        winStride = win_stride,
        padding = padding,
        scale = scale)

    for r in rects:
        upper_left = (r[0], r[1])
        lower_right = (r[0] + r[2], r[1] + r[3])
        cv2.rectangle(img, upper_left, lower_right, (0,0,255))
    
        

    
    cv2.imshow("image: %s" % (image_file,), img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



    
    


