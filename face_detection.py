import matplotlib
import numpy as np
import cv2
import matplotlib.pyplot as plt


#Loading the image to be tested
test_image = cv2.imread('binay.jpg')

#Converting to grayscale
test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

def convertToRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

haar_cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

faces_rects = haar_cascade_face.detectMultiScale(test_image_gray, scaleFactor = 1.2, minNeighbors = 5)

# Let us print the no. of faces found
print('Faces found: ', len(faces_rects))

for (x,y,w,h) in faces_rects:
     cv2.rectangle(test_image, (x, y), (x+w, y+h), (0, 0, 255), 10)

cv2.imshow('hello', convertToRGB(test_image))
#cv2.imshow('hello', test_image)

#cv2.imshow('add', test_image)
cv2.waitKey()

cv2.destroyAllWindows()



