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
    """edges = cv2.Canny(imageinzone, 100, 170)
    pyplot.subplot(121), pyplot.imshow(imageinzone, cmap='gray')
    pyplot.title('magic Card'), pyplot.xticks([]), pyplot.yticks([])
    res = cv2.resize(imageinzone, None, fx=25/17, fy=25/17, interpolation=cv2.INTER_CUBIC)
    pyplot.subplot(122), pyplot.imshow(res, cmap='gray')
    pyplot.title('resolution'), pyplot.xticks([]), pyplot.yticks([])
    pyplot.subplot(122), pyplot.imshow(edges, cmap='gray')
    pyplot.title('Edgy'), pyplot.xticks([]), pyplot.yticks([])
    pyplot.show()"""
    return compareextensionsymboltoothersymbols(imageinzone)


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
                tempresult = compareoneonone(imageinzone, extension)
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
    # TODO : function that compare and tell "CORRECT PIXELS"
    bestcorrespondancevalidpixels = 1
    # TODO : function that compare and tell "WRONG PIXELS"
    lesserrorcorrespondance = 12
    return bestcorrespondancevalidpixels - lesserrorcorrespondance


# load file here, and search for the 3d element in each line
def convertnumbertoname(name):
    print(name)
    fileopened = open("./extensionNames.txt", 'r')
    for line in fileopened:
        if name in line[0] and line[2] is not None:
            return line[2]
    return "NOT_OLD_EXTENSION"


if __name__ == "__main__":
    from sys import argv
    # TODO : SYS ARGS
    if argv.__len__() == 2:
        print(loadimage(argv[1]))
    else:
        print(loadimage("testImages/whiteUnco.png"))

