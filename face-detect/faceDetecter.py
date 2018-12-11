#!/usr/bin/python

import os
import cv2

class faceDetecter():
    def __init__(self):
        pass
    def getFaceFromCamera(self,outdir):
        camera = cv2.VideoCapture(0)
        success, image = camera.read()
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        haar = cv2.CascadeClassifier('C:\Python27\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
        faces = haar.detectMultiScale(gray_img, scaleFactor = 1.15, minNeighbors = 5, minSize = (5,5),)
        print "face count:",len(faces)
        for x, y, w, h in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('image', image)
        cv2.waitKey(0)
        camera.release()
        #cv2.destroyAllWindows()

if __name__ == '__main__':
    detecter = faceDetecter()
    detecter.getFaceFromCamera("C:\Users\dzi\Pictures")
