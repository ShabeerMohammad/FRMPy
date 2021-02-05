import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from mod_FaceID.ST_FaceID import *
from datetime import datetime as dt

from numpy import asarray
from mtcnn.mtcnn import MTCNN

import keras
import tensorflow as tf

DEBUG = True

def Logger(text):
    if DEBUG:
        dt_string = dt.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        print(dt_string + ': ' + str(text))

#Assign pre-trained mask detector model to the variable model.
cascPath = "./mod_FaceID/Models/haarcascade_frontalface_alt2.xml"
modelMaskPath = './mod_FaceID/Models/mask_recog_ver4.h5'
DATA_FACE_PATH = 'UserFaceIDs'
TRAI_EXT = 'UserFaceIDs.pkl'
FACE_RECOG_TOLERANCE = 0.60
UNKNOWN_USER_ID = 'Unknown'

print('cascPath: ',cascPath)
faceCascade = cv2.CascadeClassifier(cascPath)
faceIdentification = ST_FaceID.getInstance()
Logger('Loading faceIdentification model from: ' + DATA_FACE_PATH)
train_res = faceIdentification.Train(DATA_FACE_PATH, TRAI_EXT)
Logger(train_res)
model = load_model(modelMaskPath)

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(60, 60),flags=cv2.CASCADE_SCALE_IMAGE)
    pixels = asarray(frame)
    detector = MTCNN()
    faces = detector.detect_faces(pixels)
    print(faces)
    faces_list=[]
    preds=[]
    for face in faces:
        (x,y,w,h) = face['box']
        #Determine if face is using a mask
        face_frameColor = frame[y:y+h,x:x+w]
        face_frame = cv2.resize(face_frameColor, (224, 224))
        face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)        
        face_frame = img_to_array(face_frame)
        face_frame = np.expand_dims(face_frame, axis=0)
        face_frame =  preprocess_input(face_frame)
        #faces_list.append(face_frame)
        if len(face_frame)>0:
            preds = model.predict(face_frame)
            preds = preds>0.75
        #mask contain probabily of wearing a mask and vice versa
        for pred in preds:
        	(mask, withoutMask) = pred
        
        label = "No_Mask"
        color = (0, 0, 255)
        if mask > withoutMask:
        	label = "Mask"
        	color = (0, 255, 0)

        #If no mask Identify face
        if (label == "No_Mask") or True:
            top = 1
            right = w-1
            bottom = h-1
            left = 1
            facePositions = [(top,right,bottom ,left)]
            face_encoding = face_recognition.face_encodings(face_frameColor,facePositions)[0]
            result = faceIdentification.IdentifyFace(face_encoding,str(UNKNOWN_USER_ID),tolerance=FACE_RECOG_TOLERANCE)
            if (result['status']):
                label = result['faceID'] + ' ' +  label 

        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        cv2.putText(frame, label, (x, y- 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h),color, 2)
    
    cv2.namedWindow('Video',cv2.WINDOW_AUTOSIZE)
    frame = cv2.resize(frame, (0,0), fx=2.5, fy=2.5,interpolation = cv2.INTER_CUBIC) 
    cv2.imshow('Video', frame)
    k = cv2.waitKey(1)
    if k == 27 or k ==ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()



