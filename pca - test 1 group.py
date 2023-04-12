import zipfile
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import csv

#pull all the trainig pictures in
faces_train = {}
faces_test = {}
with zipfile.ZipFile("pic4.zip") as facezip:
    for filename in facezip.namelist():
        if not filename.endswith(".pgm") or not filename.startswith("pic4/"):
            continue 
        with facezip.open(filename) as image:
            if filename.endswith(str(10)+".pgm"):
                faces_test[filename] = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
            else:
                faces_train[filename] = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

num_pic=9

#below is for presenting the original faces: num_pic*num_pic
'''
fig, axes = plt.subplots(num_pic,num_pic,sharex=True,sharey=True,figsize=(8,8))
faceimages = list(faces_train.values())[:num_pic**2] 

for i in range(num_pic**2):
    axes[i%num_pic][i//num_pic].imshow(faceimages[i], cmap="gray")
print("Showing sample faces")
plt.show()
'''


faceshape = list(faces_train.values())[0].shape
classes = set(filename.split("/")[0] for filename in faces_train.keys())
print("Number of images:", len(faces_train))

facematrix = []
facelabel = []  #pic4
facelabel2 = [] #s1,s2,s3...
facelabel3 = [] #1.pgm, 2.pgm...

#train with sample photos
for key,val in faces_train.items():
    if val is None:
        continue
    else:
       facematrix.append(val.flatten())    
       facelabel.append(key.split("/")[0])
       facelabel2.append(key.split("/")[1])
       facelabel3.append(key.split("/")[2])
       
facematrix = np.array(facematrix)

#for i in range(len(facematrix)):
#    print(i,":  ",facematrix[i],":   ", facelabel2[i],":   ", facelabel3[i],":   ",len(facematrix[i]))

pca = PCA().fit(facematrix)
eigenfaces = pca.components_

#below is for presenting the eigenfaces: num_pic*num_pic
'''
fig, axes = plt.subplots(num_pic,num_pic,sharex=True,sharey=True,figsize=(8,8))
for i in range(num_pic**2):
    axes[i%num_pic][i//num_pic].imshow(eigenfaces[i].reshape(faceshape), cmap="gray")
print("Showing the eigenfaces")
plt.show()
'''

# Generate weights as a KxN matrix where K is the number of eigenfaces and N the number of samples
weights = eigenfaces @ (facematrix - pca.mean_).T
print("Shape of the weight matrix:", weights.shape)

fig, axes = plt.subplots(2,40,sharex=True,sharey=True,figsize=(10,8))
#axes[0][0].set_title("Query")
#axes[1][0].set_title("Best match")

#use for saving the matching results in the following step
f = open('/Users/yana/Desktop/pca', 'w')
header = ['Test person', 'Test image', 'Best match', 'Euclidean distance']
writer = csv.writer(f)
writer.writerow(header)

count=0
match=[]
#matching process
for i in faces_test.keys():
    query = faces_test[i].reshape(1,-1)
    query_weight = eigenfaces @ (query - pca.mean_).T
    euclidean_distance = np.linalg.norm(weights - query_weight, axis=0)
    best_match = np.argmin(euclidean_distance)
    
    print("Test person:",i.split("/")[1],", Test image:",i.split("/")[2], ", Best match:",facelabel2[best_match], ", Euclidean distance:",euclidean_distance[best_match])
    writer.writerow([i.split("/")[1],i.split("/")[2],facelabel2[best_match],euclidean_distance[best_match]])
    match.append(i.split("/")[1]==facelabel2[best_match])
    axes[0][count].imshow(query.reshape(faceshape), cmap="gray")
    axes[0][count].set_title(i.split("/")[1])
    axes[1][count].imshow(facematrix[best_match].reshape(faceshape), cmap="gray")
    axes[1][count].set_title(facelabel2[best_match])

    count=count+1

#show test vs best match comparison, no need to show all the time
plt.show()

f.close()
correct_count=len([x for x in match if x==True])
incorrect_count=len([x for x in match if x==False])
accuracy=correct_count/len(match)
print("Test image:",i.split("/")[2],", Count of test:",len(match),", Accuracy result:",len([x for x in match if x==True])/len(match))
