import glob
from PIL import Image
import os

count=1
for img_location in glob.glob("/Users/yana/Desktop/py/sample2/*"):
    img = Image.open(img_location)
    img.save(os.path.join(os.path.expanduser('~'),'Desktop/py/sample2',str(count)+'.pgm'))
    count=count+1
