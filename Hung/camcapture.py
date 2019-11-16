# -*- coding: utf-8 -*-

import cv2

import numpy as np
from keras.models import load_model

# Load model
my_model = load_model('model.h5')

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Crop + process captured frame
    face = frame

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Press 'c' to capture hand gesture
    if cv2.waitKey(10) & 0xFF == ord('c'):
        resizedFace = cv2.resize(face,(64,64))
        grayFace = cv2.cvtColor(resizedFace,cv2.COLOR_BGR2GRAY)
        
        # pre-process
        data = []
        data.append(grayFace.astype(float)/255.0)
        data = np.array(data)
        data = np.expand_dims(data, axis=3)
    
        # Predict
        prob = my_model.predict(data)
        pred = np.argmax(prob, axis = 1)
#        
#        # Print result
        if pred == 0:
            cv2.putText(face,'No smile',(20,20),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255))
        elif pred == 1:
            cv2.putText(face,'Smile',(20,20),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255))
        cv2.imshow('Result',face)

    # Press 'q' to exit live loop
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

## Release the capture
video_capture.release()
cv2.destroyAllWindows()

