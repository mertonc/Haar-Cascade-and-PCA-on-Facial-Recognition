import zipfile
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import csv
from imutils import build_montages
def pca_process(x):
#pull all the trainig pictures in
    faces_train = {}
    faces_test = {}
    with zipfile.ZipFile("pic4.zip") as facezip:
        for filename in facezip.namelist():
            if not filename.endswith(".pgm") or not filename.startswith("pic4/"):
                continue 
            with facezip.open(filename) as image:
                if filename.endswith(str(x)+".pgm"):
                    faces_test[filename] = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
                else:
                    faces_train[filename] = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

    num_pic=9

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
    #print("Number of images:", len(faces_train))
    
    facematrix = []
    facelabel = []  #pic4
    facelabel2 = [] #s1,s2,s3...
    facelabel3 = [] #1.pgm, 2.pgm...
    facelabel_full = []
    #train with sample photos
    for key,val in faces_train.items():
        if val is None:
            continue
        else:
           facematrix.append(val.flatten())
           facelabel_full.append(key.split("/"))
           facelabel.append(key.split("/")[0])
           facelabel2.append(key.split("/")[1])
           facelabel3.append(key.split("/")[2])
           
    facematrix = np.array(facematrix)

    pca = PCA().fit(facematrix)
    eigenfaces = pca.components_

    '''
    fig, axes = plt.subplots(num_pic,num_pic,sharex=True,sharey=True,figsize=(8,8))
    for i in range(num_pic**2):
        axes[i%num_pic][i//num_pic].imshow(eigenfaces[i].reshape(faceshape), cmap="gray")
    print("Showing the eigenfaces")
    plt.show()
    '''

    # Generate weights as a KxN matrix where K is the number of eigenfaces and N the number of samples
    weights = eigenfaces @ (facematrix - pca.mean_).T
    #print("Shape of the weight matrix:", weights.shape)

    fig, axes = plt.subplots(2,len(facelabel2)//num_pic,sharex=True,sharey=True,figsize=(10,8))

    '''f = open('/Users/yana/Desktop/book1', 'w')
    header = ['Test person', 'Test image', 'Best match', 'Euclidean distance']
    writer = csv.writer(f)
    writer.writerow(header)'''

    count=0
    match=[]
    write_contect=[]
    mismatch_names=[]
    mismatch_photos=[]
    #matching process
    for i in faces_test.keys():
        query = faces_test[i].reshape(1,-1)
        query_weight = eigenfaces @ (query - pca.mean_).T
        euclidean_distance = np.linalg.norm(weights - query_weight, axis=0)
        best_match = np.argmin(euclidean_distance)
        
        #print("Test person:",i.split("/")[1],", Test image:",i.split("/")[2], ", Best match:",facelabel2[best_match], ", Euclidean distance:",euclidean_distance[best_match])
        #writer.writerow([i.split("/")[1],i.split("/")[2],facelabel2[best_match],euclidean_distance[best_match]])
        write_contect.append([i.split("/")[1],i.split("/")[2],facelabel2[best_match],euclidean_distance[best_match]])
        if i.split("/")[1]!=facelabel2[best_match]:
           mismatch_names.append(i)
           mismatch_names.append("/".join(facelabel_full[best_match]))
           mismatch_photos.append(query.reshape(faceshape))
           mismatch_photos.append(facematrix[best_match].reshape(faceshape))
        match.append(i.split("/")[1]==facelabel2[best_match])
        axes[0][count].imshow(query.reshape(faceshape), cmap="gray")
        axes[0][count].set_title(i.split("/")[1])
        axes[1][count].imshow(facematrix[best_match].reshape(faceshape), cmap="gray")
        axes[1][count].set_title(facelabel2[best_match])

        count=count+1
        
    #show test vs best match comparison, no need to show all the time
    #plt.show()

    #f.close()
    correct_count=len([x for x in match if x==True])
    incorrect_count=len([x for x in match if x==False])
    accuracy=correct_count/len(match)
    print("Test image:",i.split("/")[2],", Count of test:",len(match),", Accuracy result:",len([x for x in match if x==True])/len(match))
    return([correct_count,incorrect_count,write_contect,mismatch_names,mismatch_photos])

correct_count=0
incorrect_count=0
f = open('/Users/yana/Desktop/pca', 'w')
header = ['Test person', 'Test image', 'Best match', 'Euclidean distance']
writer = csv.writer(f)
writer.writerow(header)

#use to collect the mismatch examples
mismatch_photos=[]
for i in range(1,11):  #the pictures are named as 1 to 10 for each person
    x=pca_process(i)
    if len(x[4])>0:
       mismatch_photos.append(x[4])
    for j in range(len(x[2])):
        writer.writerow(x[2][j])
    correct_count=x[0]+correct_count
    incorrect_count=x[1]+incorrect_count

print("Overall correct ratio: ",correct_count/(correct_count+incorrect_count))
fig, axes = plt.subplots(2,len(mismatch_photos),sharex=True,sharey=True,figsize=(10,8))
axes[0][0].set_title("Testing faces")
axes[1][0].set_title("Mismatch results")
#show testing face and the mistach face side by side
for i in range(len(mismatch_photos)):
     axes[0][i].imshow(mismatch_photos[i][0], cmap="gray")
     axes[1][i].imshow(mismatch_photos[i][1], cmap="gray")
plt.show()
