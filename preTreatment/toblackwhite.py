import cv2
im_gray = cv2.imread('test.jpg', 0)
(thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imwrite('test2.jpg', im_bw)