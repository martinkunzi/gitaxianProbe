from os import path
import cv2
import numpy
from matplotlib import pyplot


def loadimage(imagepath, zone=[177, 192, 175, 209]):
    # this part of the function tells us if we have a card, and call for the next part
    img = cv2.imread(imagepath)
    if img is not None:
        return findthesymbol(img, zone)
    else:
        return "NOT AN IMAGE"


def findthesymbol(image, zone):
    return compareextensionsymboltoothersymbols(image[zone[0]:zone[1], zone[2]:zone[3]])


def compareextensionsymboltoothersymbols(imageinzone):
    from os import listdir
    from os.path import isfile, join
    rarities = ['C', 'U', 'R', 'M']
    magicpath = "extensionsFiles/Magic/"
    result = 2000
    resultname = ""
    rarity = ''
    for rar in rarities:
        extensionpath = join(magicpath, rar)
        for f in listdir(extensionpath):
            if isfile(join(extensionpath, f)):
                extension = cv2.imread(join(extensionpath, f))
                tempresult = compareoneonone(imageinzone, extension)
                print(f + "- rarete - "+ rar +" - Result - : " + str(tempresult) + " - actualResult - :" + str(result))
                if tempresult < result:
                    result = tempresult
                    resultname = f.split('.', 1)[0]
                    rarity = rar
            # cv2.imshow(f, extension)
    if resultname is "":
        return "OLD_EXTENSION"
    else:
        resultname = convertnumbertoname(resultname)
        resultname += " -- " + converttorarity(rarity)
        return resultname


def converttorarity(rarity):
    if rarity is 'C':
        return "Common"
    elif rarity is 'U':
        return "Uncommon"
    elif rarity is 'R':
        return "Rare"
    elif rarity is 'M':
        return "Mythic"
    else:
        return "Not a reliable rarity"


def compareoneonone(imagetoprocess, extensionimage):
    # this is the comparison between the zone and the source extension image. re-scaling of expansion file done here
    # why doesn't ORB works ?
    # workaround ? Yup, it worked
    cv2.ocl.setUseOpenCL(False)
    orb = cv2.ORB_create(500, 1.6, 8, 1, 0, 2, 0, 31)

    kp1, des1 = orb.detectAndCompute(imagetoprocess, None)
    kp2, des2 = orb.detectAndCompute(extensionimage, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    datsum = 0
    arraysize = min(10, len(matches))
    if arraysize is 10:
        for i in range(0, arraysize):
            datsum += matches[i].distance
    else:
        arraysize = 1
        datsum = 200
    #uncomment those 2 for visual effect
    #result = cv2.drawMatches(imagetoprocess, kp1, extensionimage, kp2, matches[:10], None, flags=2)
    #pyplot.imshow(result), pyplot.show()
    return datsum / arraysize


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
        print(loadimage("testImages/jace_origin_mythic.png"))

