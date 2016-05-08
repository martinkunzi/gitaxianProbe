import cv2
import numpy as np
#To black and white
im_gray = cv2.imread('test.jpg', 0)
(thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#Erosion
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(im_bw,kernel,iterations = 10)
erosion = cv2.dilate(erosion,kernel,iterations = 10)

#Find contour points
im = (255-erosion) #On inverse les couleurs
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2:]
areas = [cv2.contourArea(c) for c in contours]
max_index = np.argmax(areas)
cnt=contours[max_index]

rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)
print(box)
im = cv2.drawContours(im,[box],0,(0,0,255),2)

#Calibrate
img = cv2.imread('test.jpg')
rows,cols = img.shape[:2]

# Constructiuon d'une matrice affine en lui dannat 4 points de depart et 4 points d'arrivee

# Source points
srcTri = box
# Corresponding Destination Points. Remember, both sets are of float32 type
dstTri = np.array([(cols*0.0,rows*0.33),(cols*0.85,rows*0.25), (cols*0.15,rows*0.7)],np.float32)

# Affine Transformation
warp_mat = cv2.getAffineTransform(srcTri,dstTri) # Generating affine transform matrix of size 2x3
print 'Warp mat',warp_mat
dst = cv2.warpAffine(img,warp_mat,(cols,rows)) # Now transform the image, notice dst_size=(cols,rows), not (rows,cols)
cv2.imshow('dst_rt',dst)
cv2.imwrite('final.jpg', dst)


cv2.waitKey(0)
cv2.destroyAllWindows()