import cv2
import numpy as np
from sklearn.externals import joblib
from skimage.feature import hog
from itertools import izip
from sklearn.svm import LinearSVC
import sys

__author__ = 'Quentin Jeanmonod'


# holder
class Letter:
    def __init__(self, img, x, y, w, h):
        try:
            self.img = img[y:y + h, x:x + w]
        except TypeError:
            pass
        self.x = x
        self.w = w
        self.y = y
        self.h = h


def findletters(img):
    # find card name
    img = img[18:32, 20:205]
    # make img bigger for better thinging
    img = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    # erosion (i don't know why dilate make it erode tho) to separate letters
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    # add a border so that letters that touch the sides are recognized too
    thresh = cv2.copyMakeBorder(thresh, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=255)

    letters = []

    # fiddling with contours to find the letters
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
        # ignoring small or huge things like the entire image
        if w < 5 or h < 10 or w > 50 or h > 50:
            continue
        letters.append(Letter(thresh, x, y, w, h))
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)

    # sorting the letter in order and creating a cleaned up list
    letters.sort(key=lambda l: l.x)
    newletters = []
    old = Letter(None, letters[0].x - 1, 0, 0, 0) # starting just before first letter for transform cards)
    for l in letters:
        # too much space => we finished reading the card name
        if l.x - old.x - old.w > 50:
            break
        # avoid inner contours (for o b etc)
        if old.x + old.w < l.x + l.w:
            # spaces
            if l.x - old.x - old.w > 10:
                newletters.append(Letter(thresh, old.x + old.w, 10, l.x - old.x - old.w, 30))
            # if the letter is huge, we probably have 2 of them and can cut in the middle
            # don't know how to detect 3
            if l.w > 35:
                newletters.append(Letter(thresh, l.x, l.y, l.w / 2, l.h))
                newletters.append(Letter(thresh, l.x + l.w / 2, l.y, l.w / 2, l.h))
            else:
                newletters.append(l)
            old = newletters[-1]
    
    # adding border so that all letters are the same size for classification
    maxx = 50
    maxy = 50
    for i, l in enumerate(newletters):
        difx = maxx - l.w
        dify = maxy - l.h
        marginx = difx / 2
        marginy = dify / 2
        l.img = cv2.copyMakeBorder(l.img, marginy, marginy + dify % 2, marginx, marginx + difx % 2, cv2.BORDER_CONSTANT, value=255)

    # rectangles on letters, for debug and example
    for l in newletters:
        cv2.rectangle(img, (l.x, l.y), (l.x + l.w, l.y + l.h), (0, 0, 255), 1)
    
    return newletters, img, thresh


def train():
    import json
    import urllib2
    import tqdm
    
    url = 'http://gatherer.wizards.com/Handlers/Image.ashx?multiverseid={}&type=card'
    
    # loading 1'000+ cards for training
    with open('AllSets.json', 'r') as file:
        sets = json.loads(file.read())
        cards = sets['OGW']['cards'] + sets['BFZ']['cards'] + sets['ORI']['cards'] + sets['SOI']['cards']
    
    images = []
    letters = ''
    
    # loading the image of each card on gatherer
    defect = 0
    for card in tqdm.tqdm(cards):
        id = card['multiverseid']
        name = card['name']
        
        response = urllib2.urlopen(url.format(id))
        img = np.asarray(bytearray(response.read()), dtype="uint8")
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        
        # if something goes wrong with the image or the length of the name
        # is not the same as the number of letters detected, we skip that card
        try:
            curimages, _, _ = findletters(img)
            if len(curimages) != len(name):
                defect += 1
                continue
            images[len(images):] = curimages
            letters += name
        except:
            defect += 1
            pass
    
    print('Training done, {} cards could not be read. {} % of cards successfully used for training.'.format(defect, (len(cards) - defect) * 100 / len(cards)))
    
    # training the classifier with histogram of gradients
    labels = []
    hogs = []
    for c, l in izip(letters, images):
        hogs.append(hog(l.img, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False))
        labels.append(ord(c))
    
    hog_features = np.array(hogs)
    label_features = np.array(labels)
    clf = LinearSVC()
    clf.fit(hog_features, label_features)
    joblib.dump(clf, "cls.pkl", compress=3)
    return clf


def classify(img, clf=None):
    if not clf:
        try:
            clf = joblib.load("cls.pkl")
        except:
            raise FileNotFoundException('Could not find classifier')
    
    letters, _, _ = findletters(img)
    detected = []
    for l in letters:
        roi_hog_fd = hog(l.img, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
        nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
        detected.append(nbr)
    return u''.join(chr(c) for c in detected)
    

def test(clf):
    import json
    import urllib2
    import tqdm
    import difflib
    
    url = 'http://gatherer.wizards.com/Handlers/Image.ashx?multiverseid={}&type=card'
    
    with open('AllSets.json', 'r') as file:
        sets = json.loads(file.read())
        cards = sets['C15']['cards']
        
    names = []
    detecteds = []
        
    for card in tqdm.tqdm(cards):
        id = card['multiverseid']
        name = card['name']
        
        response = urllib2.urlopen(url.format(id))
        img = np.asarray(bytearray(response.read()), dtype="uint8")
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        
        detected = classify(img, clf)
        
        detecteds.append(detected)
        names.append(name)
    
    # counting errors
    cardsum = len(names)
    cardsumcool = sum(1 if a == b else 0 for a, b in izip(names, detecteds))
    lettersum = sum(len(a) for a in names)
    lettersumcool = lettersum
    for name, detected in izip(names, detecteds):
        add = 0
        sup = 0
        for s in difflib.ndiff(name, detected):
            if s[0] == ' ':
                continue
            elif s[0] == '-':
                sup += 1
            elif s[0] == '+':
                add += 1
        lettersumcool -= sup + (add - sup)
    
    print('Checked {} cards'.format(cardsum))
    print('Estimated precision:')
    print('{} / {} ({} %) cards correct'.format(cardsumcool, cardsum, cardsumcool * 100 / cardsum))
    print('{} / {} ({} %) letters correct'.format(lettersumcool, lettersum, lettersumcool * 100 / lettersum))

    
def examplecard(img, clf):
    detected = classify(img, clf)
    print(detected)
    letters, img, thresh = findletters(img)
    for c, l in izip(detected, letters):
        cv2.putText(img, c, (l.x, l.y + l.h),cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)
        
    # making one big image instead of having dozens of small ones
    img = np.vstack((img, cv2.cvtColor(thresh[2:-2, 2:-2], cv2.COLOR_GRAY2BGR)))
    
    hogs = letters[0].img
    for l in letters[1:]:
        hogs = np.hstack((hogs, l.img))
    hogs = cv2.cvtColor(hogs, cv2.COLOR_GRAY2BGR)

    if img.shape[1] > hogs.shape[1]:
        hogs = np.hstack((hogs, np.zeros((hogs.shape[0], img.shape[1] - hogs.shape[1], 3), np.uint8)))
    else:
        img = np.hstack((img, np.zeros((img.shape[0], hogs.shape[1] - img.shape[1], 3), np.uint8)))
    
    img = np.vstack((img, hogs))
    
    cv2.imshow('{}'.format(detected), img)
    
    
def example(clf):
    paths = ['grasp.jpg', 'bastion.jpg', 'dawnbreak.jpg', 'herald.jpg', 'jace_origin_mythic.png', 'languishjpg.jpg']
    names = ['Grasp of Fate', 'Bastion Protector', 'Dawnbreak Reclaimer', 'Herald of the Host', "Jace, Prodige de Vryn", 'Languish']
    
    for name, path in izip(names, paths):
        img = cv2.imread(path)
        print(name)
        examplecard(img, clf)
        
    cv2.waitKey()
    
    
if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == 'train':
            clf = train()
        elif sys.argv[1] == 'test':
            clf = joblib.load("cls.pkl") 
            test(clf)
        else:
            print('Argument {} not recognized, use train or test'.format(sys.argv[1]))
    else:
        clf = joblib.load("cls.pkl") 
        example(clf)
