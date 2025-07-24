
import cv2
import numpy as np

filename = "../Pics/pic1.png"
img = cv2.imread(filename)

template = img[145:318,255:345,:]
cv2.imshow("template", template)

res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF)
_, max_val, _, max_loc = cv2.minMaxLoc(res)

print("max_val: ", max_val)
print("max_loc: ", max_loc)

x = max_loc[0]
y = max_loc[1]
dy = 318 - 145
dx = 345 - 255

box = np.array([(x,y), (x+dx,y), (x+dx,y+dy), (x,y+dy)])
cv2.drawContours(img, [box], 0, (0, 255, 0), 2)



cv2.imshow("img", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
