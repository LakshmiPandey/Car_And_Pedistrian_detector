import sys
print(sys.executable)

import cv2
import numpy as np


## body classifier
body_classifier = cv2.CascadeClassifier("haarcascade_fullbody.xml")


## initiate videocapture
cap = cv2.VideoCapture('walking.avi')

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.resize(frame, fx = 0.5, fy = 0.5, interpolation =cv2.INTER_LINEAR)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    body = body_classifier.detectMultiScale(gray, 1.2, 3)
    
    
    for (x, y, w, h) in body: 
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 127), 2)
        cv2.imshow("pedistrain", frame)
        
        if(cv2.waitKey(1))==13:
            break
        
        
cap.release()
cv2.destroyAllWindows()