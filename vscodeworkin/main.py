import cv2
import os
# import tensorflow as tf
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from camerasetup import CameraSetup
from connection import DbConnection
from crudoperations import UserIdsGetting
# print(cv2.__version__)
# count = cv2.cuda.getCudaEnabledDeviceCount()
# print(count)
# main area
# import tensorflow as tf
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
faceCascade= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")                 
model= cv2.dnn.readNetFromCaffe(prototxt="models/deploy.prototxt",caffeModel="models/res10_300x300_ssd_iter_140000.caffemodel")                
camerasetting=CameraSetup()
camerasetting.setupCamera(model) 

