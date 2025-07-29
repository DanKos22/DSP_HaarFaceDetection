# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 16:43:33 2025

@author: G00397054@atu.ie
"""

# -*- coding: utf-8 -*-
"""
https://stackoverflow.com/questions/36218385/parameters-of-detectmultiscale-in-opencv-using-python

@author: Dan Koskiranta
"""

import numpy as np
import cv2 as cv

face_cascade = cv.CascadeClassifier('.\HaarCascades\haarcascade_frontalface_default.xml')
#eye_cascade  = cv.CascadeClassifier('.\HaarCascades\haarcascade_eye.xml')

img  = cv.imread('flamingo-3309628_1920.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('', gray)
cv.waitKey(0)

faces = face_cascade.detectMultiScale(gray, 1.05, 5)

for (x, y, w, h) in faces:
    cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = gray[y : y + h, x : x + w]
    roi_color = img[y : y + h, x : x + w]
    #eyes = eye_cascade.detectMultiScale(roi_gray)
    #for (ex, ey, ew, eh) in eyes:
    #    cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    
cv.imshow('Image', img)
cv.waitKey(0)
cv.destroyAllWindows()