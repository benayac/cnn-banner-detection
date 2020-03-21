import cv2
import os
import time
cam = cv2.VideoCapture('00.mp4')

currentFrame = 0

while(cam.isOpened()):
    ret, frame = cam.read()
    if ret:
        cv2.imshow('Frame',frame)
    
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.imwrite(f"not_detected/{currentFrame}.jpg", frame)
            print("not detected")
            currentFrame=currentFrame+1
        elif cv2.waitKey(0) & 0xFF == ord('w'):
            cv2.imwrite(f"detected/{currentFrame}.jpg", frame)
            print("detected")
            currentFrame=currentFrame+1
        elif cv2.waitKey(0) & 0xFF == ord('s'):
            break
    else: 
        break

cam.release() 
cv2.destroyAllWindows() 