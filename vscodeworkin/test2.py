 # faceCount variable
    
    # draw boundary
    def drawBoundary(self,img,classifier,scaleFactor,minNeighbour,color,txt,imgid):
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
            cv2.putText(img,str(imgid),(x+3,y-6),cv2.FONT_HERSHEY_PLAIN,0.9,(0,250,0),1,cv2.LINE_AA)
            
            coordinates=[x,y,w,h]
        return coordinates,img

    
    
 
    # method to detect face
    def detectFace(self,img,faceClassifier,img_id,userids,cursor,clf):
        # match=self.recocgnize(img,clf,faceClassifier,1.2,10,(15,20,255))
        userid,match=self.recognize2(img,faceClassifier,1.1,10,(15,20,255))
        
        roi_img=None
        if (not match):
            coordinates,img= self.drawBoundary(img,faceClassifier,1.1,10,(105,250,10),'Unknown',img_id)
            
            print(len(coordinates))
            if (len(coordinates)>=4):
                roi_img=img[coordinates[1]:coordinates[1]+coordinates[3],coordinates[0]:coordinates[0]+coordinates[2]]
                # print(roi_img)
                user_id=5
                # generateDataset(roi_img,user_id,img_id)
                dataset=DatasetGenerator()
                dataset.generateDataset(roi_img,userid=user_id,img_id=img_id)
                # insert to db
                # query=UserInsertion()
                # query.insertUser(cursor,user_id)
            
        else:
            
            print("Match found!")
            # set the location of person here
            # query.insertUserLoc(cursor,userid,1)
                
        return roi_img,img

    # method to recognize a face
    # def matchFace(self,img,clf)
    def recocgnize(self,img,clf,classifier,scaleFactor,minNeighbour,color):
        match=False

        
        grayImg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #     detect feature in image
        features=classifier.detectMultiScale(grayImg,scaleFactor,minNeighbour)
        coordinates=[]
        for (x,y,w,h) in features:
    #         draw rectangle
            cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
            # predict
            id,result=clf.predict(grayImg[y:y+h,x:x+w])
            
            confidence=int(100*(1-result/200))
            print(id)
            if(confidence>70):
                match=True
                 
                cv2.putText(img,"user"+str(id)+"  conf: "+str(confidence),(x+2,y-5),cv2.FONT_HERSHEY_COMPLEX,0.8,color,1,cv2.LINE_AA)
   

        
        return match


          
                # cv2.putText(img, text=str(round(face_confidence, 1))+txt, org=(x1, y1-15), 
                #     fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.6,
                #     color=color, thickness=1)
    
    # 2nd recognize func
    def recognize2(self,img,classifier,scaleFactor,minNeighbour,color):
        match=False
        grayImg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        grayImg=np.array(img,'uint8')
    #       
        id=0
        # detect feature in image
        features=classifier.detectMultiScale(img,scaleFactor,minNeighbour)

        
        ids=[]
        for (x,y,w,h) in features:
            df=DeepFace.find(img_path=grayImg[y:y+h,x:x+w],db_path="images", enforce_detection=False,model_name='Facenet',model=self.model,distance_metric='cosine',detector_backend='opencv')
            print("++++++++++++++++++++")
            print(x,y,w,h)
            
            # cv2.rectangle(img, (x,y),(x+w,y+h), color=color, thickness=2)
            # grayImg=img[y:y+h,x:x+w]
            try:
                   
                
                # print(df.shape[0])
                
                if (df.shape[0]>0):
                    
                    matched=df.iloc[0].identity
                    # cosineval=df.iloc[0].Facenet_cosine
                    cosineval=df.iloc[0].Facenet_cosine
                    confidence=int((1-cosineval)*100)
                    matched=str(matched)
                    id=int(matched.split("\\")[1].split('/')[0].split('_')[1])
                    ids.append(id)
                    print(df.head(2))
                    if(confidence>65):
                        match=True
                        # cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
                        cv2.rectangle(img, (x,y),(x+w,y+h), color=color, thickness=2)
                        cv2.putText(img,"user"+str(id)+ " conf: "+str(confidence),(x+2,y-5),cv2.FONT_HERSHEY_COMPLEX,0.8,color,1,cv2.LINE_AA)
                    # else:
                    #     cv2.rectangle(img, (x,y),(x+w,y+h), color=(200,100,100), thickness=2)
                        
                else:
                    match=False  #     match=False
            except:
                print('Something went wrong!')  
                    
            

        return ids,match