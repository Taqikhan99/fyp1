# from deepface import DeepFace
# DeepFace.stream("images",enable_face_analysis=False,time_threshold=1,frame_threshold=1)
import cv2
import threading
import tensorflow as tf
import time
from deepface import DeepFace
from deepface.basemodels import VGGFace,Facenet,OpenFace,FbDeepFace
model=Facenet.loadModel()
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


if tf.test.gpu_device_name(): 

    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

else:

   print("Please install GPU version of TF")
class camThread(threading.Thread):
    
    def __init__(self, previewName, camID):
        threading.Thread.__init__(self)
        self.previewName = previewName
        self.camID = camID
    def run(self):
        print ("Starting " + self.previewName)
        
        camPreview(self.previewName, self.camID)

def camPreview(previewName, camID):
    cv2.namedWindow(previewName)
    cam = cv2.VideoCapture(camID)
    if cam.isOpened():  # try to get the first frame
        rval, frame = cam.read()
    else:
        rval = False
    ids=[]
    while rval:
        
        cv2.imshow(previewName, frame)
        rval, frame = cam.read()
        
        key = cv2.waitKey(10)
        if key == 27:  # exit on ESC
            break
    cv2.destroyWindow(previewName)

# Create two threads as follows

thread1 = camThread("Camera 1", 0)
thread2 = camThread("Camera 2", 1)
thread1.start()
thread2.start()


