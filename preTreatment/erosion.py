import cv2
import numpy as np

img = cv2.imread('test2.jpg',0)
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(img,kernel,iterations = 10)
erosion = cv2.dilate(erosion,kernel,iterations = 10)

cv2.imshow('img',img)
cv2.imshow('erosion',erosion)
cv2.imwrite('test3.jpg', erosion)

cv2.waitKey(0)
cv2.destroyAllWindows()