import cv2
import numpy as np
import matplotlib.pyplot as plt

def toblackandwhite(im):
    #To black and white
    im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    (thresh, im_bw) = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = np.ones((5,5),np.uint8)
    kernel = np.ones((5,5),np.uint8)
    return im_bw,kernel

def erodeAndDilate(im_bw,kernel):
    #Erosion et dillatation
    erosion = cv2.erode(im_bw,kernel,iterations = 10)
    erosion = cv2.dilate(erosion,kernel,iterations = 10)
    return erosion

def findcontourPoints(im):
    im2 = im
    im_bw,kernel = toblackandwhite(im)
    erosion = erodeAndDilate(im_bw,kernel)
    erosion = (255-erosion) #On inverse les couleurs
    contours, hierarchy = cv2.findContours(erosion,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2:]
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt=contours[max_index]

    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    im2 = cv2.drawContours(im2,[box],0,(0,0,255),2)

    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("Show",im2)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return box



def pretreat(imagepath):
    im = cv2.imread(imagepath)
    box = findcontourPoints(im)


    img = cv2.imread(imagepath)
    rows,cols = img.shape[:2]

    # Constructiuon d'une matrice affine en lui dannat 4 points de depart et 4 points d'arrivee

    # Source points
    rows,cols,ch = img.shape
    #box[0][0], box[1][0] = box[1][0], box[0][0]
    #box[0][1], box[1][1] = box[1][1], box[0][1]
    box[1][0], box[3][0] = box[3][0], box[1][0]
    box[1][1], box[3][1] = box[3][1], box[1][1]

    box[0][0], box[2][0] = box[2][0], box[0][0]
    box[0][1], box[2][1] = box[2][1], box[0][1]

    box[2][0], box[3][0] = box[3][0], box[2][0]
    box[2][1], box[3][1] = box[3][1], box[2][1]
    print(box)
    pts1 = np.float32(box)
    #pts1 = np.float32([[300,200],[650,430],[12,690],[360,880]])
    pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])

    M = cv2.getPerspectiveTransform(pts1,pts2)

    dst = cv2.warpPerspective(img,M,(300,300))
    plt.subplot(121),plt.imshow(img),plt.title('Input')
    plt.subplot(122),plt.imshow(dst),plt.title('Output')
    plt.show()


if __name__ == "__main__":
    from sys import argv
    if len(argv) == 2:
        pretreat(argv[1])
    else:
        print("La syntaxe est \"pretreat.py file.jpg\"")