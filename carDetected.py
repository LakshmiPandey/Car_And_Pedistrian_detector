import sys
print(sys.executable)

import cv2
import numpy as np


## car  classifier
car_classifier = cv2.CascadeClassifier("haarcascade_car")


## initiate videocapture
cap = cv2.VideoCapture('walking.avi')

while cap.isOpened():
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_classifier.detectMultiScale(gray, 1.3, 3)
    
    
    for (x, y, w, h) in cars: 
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 127), 2)
        cv2.imshow("cars detected", frame)
        
        if(cv2.waitKey(1))==13:
            break
        
        
cap.release()
cv2.destroyAllWindows()