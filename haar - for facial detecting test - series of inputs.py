import csv
import imutils
import cv2
import os
import glob
from PIL import Image
import numpy as np
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
from imutils import build_montages
import math

def test(scale_factor,min_neighbors):
    count=0
    detect_face=0
    count_face=0
    faces_collect = []

    for img_location in glob.glob("/Users/yana/Desktop/py/sample/*.pgm"):
        #print(img_location)
        image = cv2.imread(img_location)
        image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        rects = detector.detectMultiScale(gray, scaleFactor=scale_factor,
                minNeighbors=min_neighbors, minSize=(10, 10),
                flags=cv2.CASCADE_SCALE_IMAGE)
        #print("{} faces detected...".format(len(rects)))
        count=count+1
        count_face=count_face+len(rects)

        if len(rects)==0:
            continue
        else:
            detect_face=detect_face+1
            for (x, y, w, h) in rects:    
                
                crop_img = image[y:y+h+20, x:x+w+20]
                crop_img = imutils.resize(crop_img, width=95,height=105)
                faces_collect.append(crop_img)
                #cv2.imshow("cropped", crop_img)
            for (x, y, w, h) in rects:    
                    
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
            if not cv2.imwrite(os.path.join(os.path.expanduser('~'),'Desktop/py/sample','s'+str(count)+'.png'), image):
                raise Exception
    print("scale_factor: ",scale_factor,",scale_factor: ",min_neighbors)
    print("sample count:",count,",how many contain faces:",detect_face,",how many faces were detacted",count_face)
    return (scale_factor,min_neighbors,count,detect_face,count_face)
    '''im_shape = (249,296)
    montage_shape = (10,max(math.ceil(count_face/10),5))
    montages=build_montages(faces_collect, im_shape, montage_shape)
    for montage in montages:
        cv2.imshow("Montage", montage)'''

f = open('/Users/yana/Desktop/haar', 'w')
header = ['scale factor', 'min neighbors', 'sample count', 'how many contain faces','how many faces were detacted']
writer = csv.writer(f)
writer.writerow(header)
for i in range(5,26):
    for j in np.arange(1.01,1.21,0.01):
        writer.writerow(test(j,i))
