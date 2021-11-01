import numpy as np
from PIL import Image
import os
import cv2


def trainClassifier(data_dir):
    # path = [os.path.join(data_dir,f) for f in os.listdir(data_dir)]
    # subfolders = [ f.path for f in os.scandir(data_dir) if f.is_dir() ]
    # print(subfolders)
    

    # ids=np.array(ids)

    

    usernames=[]
    paths=[]

    # getting subfolders in data folder
    for users in os.listdir(data_dir):
        usernames.append(users)
        print(usernames)

    for user in usernames:
        for image in os.listdir(data_dir+"\{}".format(user)):
            path_image=os.path.join(data_dir+"\{}".format(user),image)

            # data/user_id/image
            paths.append(path_image)

    print(paths)
    faces=[]
    ids=[]


    for imagepath in paths:
        img=Image.open (imagepath)
        imageNp=np.array(img,'uint8')
        # extracting user id from file path
        id=int(imagepath.split("\\")[2].split("_")[1].split(".")[0])
        print(id)
        faces.append(imageNp)
        ids.append(id)
    
    ids=np.array(ids)

    clf=cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces,ids)
    clf.write("classifier.yml")


trainClassifier("data")

