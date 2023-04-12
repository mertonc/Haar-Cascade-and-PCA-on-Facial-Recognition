
import imutils
import cv2
import os
import glob
from PIL import Image
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

count=1
for img_location in glob.glob("/Users/yana/Desktop/py/sample5/*.pgm"):
    print(img_location)
    image = cv2.imread(img_location)
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #detection process
    rects = detector.detectMultiScale(gray, scaleFactor=1.05,
            minNeighbors=16, minSize=(20, 30),
            flags=cv2.CASCADE_SCALE_IMAGE)
    print("{} faces detected...".format(len(rects)))

    if len(rects)==0:
        continue
    #crop the recognized faces and save them under a certain place
    else:
        for (x, y, w, h) in rects:    
            #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            crop_img = image[y:y+h+20, x:x+w+20]
            crop_img=cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            crop_img = imutils.resize(crop_img, width=95,height=105)
        #cv2.imshow("cropped", crop_img)
        if not cv2.imwrite(os.path.join(os.path.expanduser('~'),'Desktop/py/pic3/s45',str(count)+'.png'), crop_img):
             raise Exception
        count=count+1
 #loop through the photos and change the format, then delete the old ones
for img_location in glob.glob("/Users/yana/Desktop/py/pic3/s45/*.png"):
    img = Image.open(img_location)
    img.save(img_location.replace("png", "pgm"))
    os.remove(img_location)
