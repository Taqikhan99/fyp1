import numpy as np
from PIL import Image
import os
import cv2


def trainClassifier(data_dir):

    usernames=[]
    paths=[]

    for root,dirs,files in os.walk(data_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                path=os.path.join(root,file)
                label=os.path.basename(root).replace(" ","-").lower()
                # print(label,path)
                paths.append(path)
                pilImage=Image.open(path).convert("L") #convert to grayscale
                imageArray= np.array(pilImage,'uint8')

     
    faces=[]
    ids=[]


    for imagepath in paths:
        print(imagepath)
        img=Image.open (imagepath)
        imageNp=np.array(img,'uint8')
        # extracting user id from file path
        id=int(imagepath.split("\\")[2].split("_")[1].split(".")[0])
        faces.append(imageNp)
        ids.append(id)
    
    ids=np.array(ids)
    # print(faces)
    # print(ids)
    clf=cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces,ids)
    clf.write("classifier.yml")
    print("Training successful!")    


trainClassifier("images")

