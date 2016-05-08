import cv2
import numpy as np
import matplotlib.pyplot as plt

#To black and white
im_gray = cv2.imread('test.jpg', 0)
(thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
kernel = np.ones((5,5),np.uint8)
#Erosion et dillatation
erosion = cv2.erode(im_bw,kernel,iterations = 10)
erosion = cv2.dilate(erosion,kernel,iterations = 10)
#Find contour points
im = (255-erosion) #On inverse les couleurs
contours, hierarchy = cv2.findContours(im,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2:]
areas = [cv2.contourArea(c) for c in contours]
max_index = np.argmax(areas)
cnt=contours[max_index]

#Calibrate
rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)
print(box)
im = cv2.drawContours(im,[box],0,(0,0,255),2)

img = cv2.imread('test.jpg')
rows,cols = img.shape[:2]

# Constructiuon d'une matrice affine en lui dannat 4 points de depart et 4 points d'arrivee

# Source points
rows,cols,ch = img.shape

pts1 = np.float32(box)
pts1 = np.float32([[300,200],[650,430],[12,690],[360,880]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])

M = cv2.getPerspectiveTransform(pts1,pts2)

dst = cv2.warpPerspective(img,M,(300,300))
plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()
