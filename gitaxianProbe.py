"""
Welp, it will call every part of the program, one after another.
"""
import cv2

if __name__ == "__main__":
    from system import argv
    if len(argv) is 2:
        redressedimage = bastienmain(argv[1])
        imagename = quentinmain(redressedimage, clf)
		imageextension = martinmain(redressedimage)
		cv2.showimage(imagename + "," + imageextension, redressedimage)
		cv2.waitforkey(0)
	else:
	    print("number of arguments incorrect. Should only be a path to an image")