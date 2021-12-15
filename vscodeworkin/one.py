
import cv2
import os
from pathlib import Path

from connection import personData
from connection import cursor

try:
    print(personData)
    personIds=[]
    personNames=[]

    for row in range(len(personData)):
        # personIds.append(row[0])
        # personNames.append(row[1])
        personIds.append( personData[row][0])
        personNames.append( personData[row][1])
    print(personIds[-1]+1)
except:
    print("Something went wrong!")


def generateDataset(img, userid,img_id):

    # creating a folder for a user
    Path("data/user_{}".format(userid)).mkdir(parents=True,exist_ok=True)

    # convert to gray and resize before saving
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray=cv2.resize(gray,(200,200))

    # writing images in the user folder
    savingPath="data/user_"+str(userid)+"/user_"+str(userid)+"."+str(img_id)+".jpg"
    cv2.imwrite(savingPath,gray)
    print("Images recorded successfully!")


    

# draw boundary
def drawBoundary(img,classifier,scaleFactor,minNeighbour,color,txt):
#     convert to gray
    grayImg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     detect feature in image
    features=classifier.detectMultiScale(grayImg,scaleFactor,minNeighbour)
    coordinates=[]
    for (x,y,w,h) in features:
#         draw rectangle
        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)

        #label rectangle
        cv2.putText(img,txt,(x+3,y-3),cv2.FONT_HERSHEY_PLAIN,0.9,color,1,cv2.LINE_AA)
        
        coordinates=[x,y,w,h]
    return coordinates,img


# method to recognize a face
def recocgnize(img,clf,classifier,scaleFactor,minNeighbour,color):
    match=False
    grayImg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     detect feature in image
    features=classifier.detectMultiScale(grayImg,scaleFactor,minNeighbour)
    coordinates=[]
    for (x,y,w,h) in features:
#         draw rectangle
        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)

        # predict
        id,_=clf.predict(grayImg[y:y+h,x:x+w])
        if id==1:
            print(id)
            match=True
            cv2.putText(img,"Taqi",(x+2,y-5),cv2.FONT_HERSHEY_COMPLEX,0.8,color,1,cv2.LINE_AA)
        elif id==2:
            print(id)
            match=True
            cv2.putText(img,"Ammad",(x+2,y-5),cv2.FONT_HERSHEY_COMPLEX,0.8,color,1,cv2.LINE_AA)
        # elif id==3:
        #     match=True
        #     cv2.putText(img,"Afridi",(x+2,y-5),cv2.FONT_HERSHEY_COMPLEX,0.8,color,1,cv2.LINE_AA)

        #label rectangle
        # cv2.putText(img,txt,(x+3,y-3),cv2.FONT_HERSHEY_PLAIN,0.9,color,1,cv2.LINE_AA)
        
        coordinates=[x,y,w,h]
    
    
    return img,match

# method to detect face
def detectFace(img,faceClassifier,img_id,clf):
    Rimg,match=recocgnize(img,clf,faceCascade,1.1,6,(185,200,120))
    
    if (not match):
        coordinates,img= drawBoundary(img,faceClassifier,1.1,6,(105,250,10),'My Face')
        print(len(coordinates))
        if (len(coordinates)==4):
            roi_img=img[coordinates[1]:coordinates[1]+coordinates[3],coordinates[0]:coordinates[0]+coordinates[2]]
            print(roi_img)
            user_id=personIds[-1]+1
            generateDataset(roi_img,user_id,img_id)
            # try:

            #     cursor.execute('''
            #             INSERT INTO tbPerson (personId, pName, imagesPath)
            #             VALUES
            #             (?,'unknown',?)
                        
            #             ''',user_id,'data/user_'+str(user_id))
            #     cursor.commit()
            #     cursor.close()
            #     print('Saved to database')
            # except:
            #     print('Already saved for this user')
            
    else:
        print("Match found!")
        
            
    return img



# main

faceCascade= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

clf=cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.yml")

video_capture=cv2.VideoCapture(0);
# addr="http://192.168.2.104:8080/video"
video_capture.set(10,150)
# video_capture.open(addr)
img_id=0

while True:
    ret,img=video_capture.read()
    print(ret)
    img=detectFace(img,faceCascade,img_id,clf)

    cv2.imshow("Face detection",img)
    
    img_id+=1
    if cv2.waitKey(1) & 0xFF == ord('z'):
        
        break
# release webcam
video_capture.release()
cv2.destroyAllWindows()