import cv2
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

model = tf.keras.models.load_model('banner_detection_model')
label = ['detected', 'not detected']

cam = cv2.VideoCapture('00.mp4')
currentFrame = 0
while(cam.isOpened()):
    ret, frame = cam.read()
    if ret:
        #predict frame
        img = Image.fromarray(frame)
        b, g, r = img.split()
        img = Image.merge("RGB", (r, g, b))
        img = img.resize((150, 150))
        img = np.expand_dims(img, axis=0)
        result = model.predict_classes(img/255.)
        print(result[0][0])
        cv2.waitKey(0)
        if result[0][0] == 0:
            print(label[result[0][0]], currentFrame)
            cv2.putText(frame, label[result[0][0]], (360, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), thickness=3)
            cv2.imshow('Frame', frame)
            currentFrame+=1
        elif result[0][0] != 0:
            print(label[result[0][0]], currentFrame)
            cv2.putText(frame, label[result[0][0]], (360, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), thickness=3)
            cv2.imshow('Frame', frame)
            currentFrame+=1
    else: 
        break

cam.release() 
cv2.destroyAllWindows() 