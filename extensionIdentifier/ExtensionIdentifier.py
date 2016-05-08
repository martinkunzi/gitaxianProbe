from os import path
import cv2
import numpy
from matplotlib import pyplot


def loadimage(imagepath, zone=[177, 191, 175, 208]):
    # this part of the function tells us if we have a card, and call for the next part
    img = cv2.imread(imagepath)
    if img is not None:
        pyplot.subplot(121), pyplot.imshow(img, cmap='gray')
        pyplot.title('magic Card'), pyplot.xticks([]), pyplot.yticks([])
        pyplot.show()
        return findthesymbol(img, zone)
    else:
        return "NOT AN IMAGE"


def findthesymbol(image, zone):

    imageinzone = image[zone[0]:zone[1], zone[2]:zone[3]]
    imageingrayzone = cv2.cvtColor(imageinzone, cv2.COLOR_BGR2GRAY)
    graymage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return compareextensionsymboltoothersymbols(graymage)


def compareextensionsymboltoothersymbols(imageinzone):
    from os import listdir
    from os.path import isfile, join
    rarities = ['C', 'U', 'R', 'M']
    magicpath = "extensionsFiles/Magic/"
    result = 0
    resultname = ""
    rarity = ''
    for rar in rarities:
        extensionpath = join(magicpath, rar)
        for f in listdir(extensionpath):
            if isfile(join(extensionpath, f)):
                extension = cv2.imread(join(extensionpath, f))
                grayxtension = cv2.cvtColor(extension, cv2.COLOR_BGR2GRAY)
                tempresult = compareoneonone(imageinzone, grayxtension)
                if tempresult > result:
                    result = tempresult
                    resultname = f.split(str=".")[0]
                    rarity = rar
                    print(resultname)
            # cv2.imshow(f, extension)
    if resultname is "":
        return "OLD_EXTENSION"
    else:
        resultname = convertnumbertoname(resultname)
        resultname += " -- " + rarity
        return resultname


def compareoneonone(imagetoprocess, extensionimage):
    # this is the comparison between the zone and the source extension image. re-scaling of expansion file done here
    # why doesn't ORB works ?
    # workaround ? Yup, it worked
    cv2.ocl.setUseOpenCL(False)
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(extensionimage, None)
    kp2, des2 = orb.detectAndCompute(extensionimage, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)
    print(bf)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    print len(matches)
    # result = cv2.drawMatches(extensionimage, kp1, extensionimage, kp2, matches[:10], None, flags=2)
    # pyplot.imshow(result), pyplot.show()

    return len(matches)


def convertnumbertoname(name):
    fileopened = open("./extensionNames.txt", 'r')
    for line in fileopened:
        if name in line.split()[0] and line.split()[2] is not None:
            return line.split()[2]
    return "NOT_OLD_EXTENSION"


def martinpart(pathtoimage):
    print loadimage(pathtoimage)


if __name__ == "__main__":
    from sys import argv
    if argv.__len__() == 2:
        print(loadimage(argv[1]))
    else:
        print(loadimage("testImages/whiteCommon.jpg"))

