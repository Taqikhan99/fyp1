import time
from UserFace import Vision
import cv2
import os
from threading import Thread
from connection import DbConnection
from crudoperations import UserIdsGetting
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
# making database connection 
conn=DbConnection()
conn.connectToDb()

# getting userIds
users=UserIdsGetting()
userids=users.gettingUserId(conn.cursor)
userLocs=users.getUserLocations(conn.cursor)

print(userids)
# print(userLocs)
# clf=cv2.face.LBPHFaceRecognizer_create()
# clf.read("classifier.yml")


class CameraSetup:
    prevTime=0
    newTime=0
    def fpsCalculate(self):
        newTime=time.time()
        fps=1/(newTime-CameraSetup.prevTime)
        CameraSetup.prevTime=newTime
        fps=str(int(fps))
        return fps
    def setupCamera(self,model):
        video_capture=cv2.VideoCapture(0);
        # addr="http://192.168.2.104:8080/video"
        # video_capture.set(10,150)
        # video_capture.open(addr)
       
        vision1=Vision()

        while video_capture.isOpened():
            ret,img=video_capture.read()
           
            # roiImg,img=vision1.detectFace(img,model,img_id,userids,conn.getCursor(),clf)
            # roiImg,img=vision1.detectFaceDNN(img,model,0.5,img_id,userids,conn.getCursor(),clf,userLocs)
            img=vision1.recognize3(img,model,userids,conn.getCursor(),1.1,2,(0,255,0),userLocs)
           

            fps=self.fpsCalculate()
            cv2.putText(img,str(int(fps)), (7, 70),cv2.FONT_HERSHEY_COMPLEX,1, (100, 255, 0),2, cv2.LINE_AA)
            
            
            cv2.namedWindow("Face detection",cv2.WINDOW_NORMAL)
            cv2.imshow("Face detection",img)
            
            
                # print(img_id)
            if cv2.waitKey(1) & 0xFF == ord('z'):
                break
            
        # release webcam
        video_capture.release()
        cv2.destroyAllWindows()

# model= cv2.dnn.readNetFromCaffe(prototxt="models/deploy.prototxt",caffeModel="models/res10_300x300_ssd_iter_140000.caffemodel") 

# class WebCamsetup:
#     prevTime=0
#     newTime=0
#     # init method
#     def __init__(self,camid=0):
#         self.camid=camid

#         self.vcap=cv2.VideoCapture(self.camid)

#         if self.vcap.isOpened() is False :
#             print("[Exiting]: Error accessing webcam stream.")
#             exit(0)
#         fps_input_stream = int(self.vcap.get(5)) # hardware fps
#         print("FPS of input stream: {}".format(fps_input_stream))
            
#         # reading a single frame from vcap stream for initializing 
#         self.grabbed , self.frame = self.vcap.read()
#         if self.grabbed is False :
#             print('[Exiting] No more frames to read')
#             exit(0)        # self.stopped is initialized to False 
#         self.stopped = True        # thread instantiation  
#         self.t = Thread(target=self.update, args=())
#         self.t.daemon = True # daemon threads run in background 

#     # method to start thread 
#     def start(self):
#         self.stopped = False
#         self.t.start()
#     # method passed to thread to read next available frame  
#     def update(self):
#         while True :
#             if self.stopped is True :
#                 break
#             self.grabbed , self.frame = self.vcap.read()
#             if self.grabbed is False :
#                 print('[Exiting] No more frames to read')
#                 self.stopped = True
#                 break 
#         self.vcap.release()

#     # method to return latest read frame 
#     def read(self):
#         fps=self.fpsCalculate()
#         cv2.putText(self.frame,str(int(fps)), (7, 70),cv2.FONT_HERSHEY_COMPLEX,1, (100, 255, 0),2, cv2.LINE_AA)
#         return self.frame    # method to stop reading frames 
#     def stop(self):
#         self.stopped = True
   
#     def fpsCalculate(self):
#         newTime=time.time()
#         fps=1/(newTime-WebCamsetup.prevTime)
#         WebCamsetup.prevTime=newTime
#         fps=str(int(fps))
#         return fps

# webcam_stream = WebCamsetup(camid=1) # 0 id for main camera
# webcam_stream.start()# processing frames in input stream
# num_frames_processed = 0 
# start = time.time()
# vision1=Vision()
# while True :
#     if webcam_stream.stopped is True :
#         break
#     else :
        
#         frame = webcam_stream.read()    # adding a delay for simulating video processing time 
#     delay = 0.01 # delay value in seconds
#     time.sleep(delay) 
#     img=vision1.recognize3(frame,model,userids,conn.getCursor(),1.1,2,(0,255,0),userLocs)
#     num_frames_processed += 1    # displaying frame 
#     cv2.imshow('frame' , frame)
#     key = cv2.waitKey(1)
#     if key == ord('q'):
#         break
# end = time.time()
# webcam_stream.stop() # stop the webcam stream
# # printing time elapsed and fps 
# elapsed = end-start
# fps = num_frames_processed/elapsed 
# print("FPS: {} , Elapsed Time: {} ".format(fps, elapsed))# closing all windows 
# cv2.destroyAllWindows()