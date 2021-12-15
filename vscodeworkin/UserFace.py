import cv2
import os
from pathlib import Path
from deepface import DeepFace
from deepface.basemodels import VGGFace,Facenet,OpenFace,FbDeepFace
from crudoperations import UserInsertion
import numpy as np
import time
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1"


query=UserInsertion()
class Vision:
    # model=OpenFace.loadModel()
    model=Facenet.loadModel()
    # model=VGGFace.loadModel()
    imgId=1


    def recognize3(self,img,classifier,userids,cursor,scaleFactor,minNeighbour,color,userlocs):
        cursor=cursor
        match=False
        
        # Get the height and width of the input image.
        image_height, image_width, _ = img.shape
        
        preprocessed_image = cv2.dnn.blobFromImage(img, scalefactor=scaleFactor, size=(200, 200), swapRB=False, crop=False)
        # Set the input value for the model.
        classifier.setInput(preprocessed_image)
        # Perform the face detection on the image.
        results = classifier.forward()
        x1,y1,x2,y2=self.drawBoundary2(img,classifier,scaleFactor,0.6)
        
        id=0
        # print(results)
        ids=[]
        faceCount=0
         # Loop through each face detected in the image.
        for face in results[0][0]:
             
            # Retrieve the face detection confidence score.
            face_confidence = face[2]
            # print(face_confidence)
            # Check if the face detection confidence score is greater than the thresold.
            if face_confidence > 0.4:
                faceCount+=1 
                cv2.putText(img,"Face Count: "+str(faceCount),(200,10),cv2.FONT_HERSHEY_PLAIN,0.9,((255,255,255)),1,cv2.LINE_AA)  
                # Retrieve the bounding box of the face.
                bbox = face[3:]
                
                # Retrieve the bounding box coordinates of the face and scale them according to the original size of the image.
                x1 = int(bbox[0] * image_width)
                y1 = int(bbox[1] * image_height)
                x2 = int(bbox[2] * image_width)
                y2 = int(bbox[3] * image_height)
                roiImage=img[y1:y2,x1:x2]

                try:
                    images=[(1,2,34,4,5)]
                    df=DeepFace.find(img_path=roiImage,db_path="images", enforce_detection=False,model_name='Facenet',model=self.model,distance_metric='cosine',detector_backend='opencv')
                    
                    print(df.head())
                    
                    if (df.shape[0]>0):
                        matched=df.iloc[0].identity
                        # cosineval=df.iloc[0].Facenet_cosine
                        cosineval=df.iloc[0].Facenet_cosine
                        confidence=int((1-cosineval)*100)
                        matched=str(matched)
                        id=int(matched.split("\\")[1].split('/')[0].split('_')[1])
                        ids.append(id)
                        
                        
                        if(confidence>60):
                            match=True
                                # cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
                            cv2.rectangle(img, pt1=(x1 , y1), pt2=(x2, y2), color=color, thickness=2)
                            cv2.putText(img,"user"+str(id)+ " conf: "+str(confidence),(x1+2,y1-5),cv2.FONT_HERSHEY_COMPLEX,0.6,color,1,cv2.LINE_AA)

                            # running query every 10 sec to insert location of user
                            query.insertUserLoc(cursor,ids,1,userlocs)
                    

                    else:
                        # new face found
                        dataset=DatasetGenerator()
                        cv2.rectangle(img, pt1=(x1 , y1), pt2=(x2, y2), color=((255,255,0)), thickness=2)
                        cv2.putText(img,"New Face Detected",(10,10),cv2.FONT_HERSHEY_PLAIN,0.9,((255,255,0)),1,cv2.LINE_AA)

                        
                        user_id=userids[-1]+1

                        
                        Vision.imgId+=1
                        # print("User id: ",user_id)
                        # print("img id: ",Vision.imgId)   
                            # generating data set
                        dataset.generateDataset(roiImage,userid=user_id,img_id=Vision.imgId)
                        UserInsertion.insertUser(self,cursor,user_id)
                        match=False
                except Exception as e:
                    print("Some error happened! ",e)

        return img
    def drawBoundary2(self,img,model,scalefactor,minConf):
        roiImage = img.copy()
        x1,y1,x2,y2=0,0,0,0
        #     convert to gray
        grayImg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Get the height and width of the input image.
        image_height, image_width, _ = img.shape
        
        preprocessed_image = cv2.dnn.blobFromImage(img, scalefactor=scalefactor, size=(100, 100), swapRB=False, crop=False)
        # Set the input value for the model.
        model.setInput(preprocessed_image)
        # Perform the face detection on the image.
        results = model.forward()
        
        # Loop through each face detected in the image.
        for face in results[0][0]:
            
            # Retrieve the face detection confidence score.
            face_confidence = face[2]
            # print(face_confidence)
            # Check if the face detection confidence score is greater than the thresold.
            if face_confidence > minConf:
                # Retrieve the bounding box of the face.
                bbox = face[3:]
                # print(bbox)
                # Retrieve the bounding box coordinates of the face and scale them according to the original size of the image.
                x1 = int(bbox[0] * image_width)
                y1 = int(bbox[1] * image_height)
                x2 = int(bbox[2] * image_width)
                y2 = int(bbox[3] * image_height)
                # roiImage=img[y1:y2,x1:x2]
                
    
                # cv2.putText(img, text=str(round(face_confidence, 1))+txt, org=(x1, y1-15), 
                #     fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.6,
                #     color=color, thickness=1)
                
        return x1,y1,x2,y2
        
    def detectFaceDNN(self,img,model,minConf,img_id,userids,cursor,clf,userLocs):
        roiImage=None
        ids,match=self.recognize3(img,model,1.1,6,color=(0,255,0))
        dataset=DatasetGenerator()
        # roi_img=None
        if (not match):

            
            x1,y1,x2,y2=self.drawBoundary2(img,model,1.1,minConf)
            roiImage=img[y1:y2,x1:x2]
            # Draw a bounding box around a face on the copy of the image using the retrieved coordinates.
            cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=(255,0,0), thickness=2)
            user_id=5
                # generateDataset(roi_img,user_id,img_id)
            if(roiImage is not None):
                print('New Face!')
                dataset.generateDataset(roiImage,userid=user_id,img_id=img_id)
        else:
            print("Match found!")
            # set the location of person here
            
            # query.insertUserLoc(cursor,ids,1,userLocs)
        return roiImage,img

class DatasetGenerator:
    second=1
    def convertToGray(self,img):
        # convert to gray and resize before saving
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray=self.resizeImg(gray)
        return gray
    def resizeImg(self,img):
        resize=cv2.resize(img,(200,200))
        return resize
    def generateDataset(self,img, userid,img_id):

        # creating a folder for a user
        Path("images/user_{}".format(userid)).mkdir(parents=True,exist_ok=True)
        # gray=self.convertToGray(img)
        # writing images in the user folder
        try:
            if DatasetGenerator.second<=3:
                
                img=self.convertToGray(img)
                img=self.resizeImg(img)
                savingPath="images/user_"+str(userid)+"/user_"+str(userid)+"."+str(img_id)+".jpg"
                cv2.imwrite(savingPath,img)
                DatasetGenerator.second+=1
            else:
                print("images saved")
                return
            
                
        except:
            print('Check please')



    
