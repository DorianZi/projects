#!/usr/bin/python

import os
import cv2
import numpy as np
from optparse import OptionParser

OWNERS = ["dorian","vanny"]


class faceDetecter():
    def __init__(self,owner,picDir):
        self.pic_repo = picDir
        self.owner = owner
        pass
        
    def catchFacesFromCamera(self,mode="collect"):	
        if not self.owner in OWNERS:
            os.exit("Error: owner name unknown!")
        camera = cv2.VideoCapture(0)
        haar = cv2.CascadeClassifier('C:\Python27\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
        count = 0
        for n in range(30):
            success, image = camera.read()
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = haar.detectMultiScale(gray_img, scaleFactor = 1.15, minNeighbors = 5, minSize = (5,5))
            if not len(faces):
                print "No face detected!"
            if mode == "detect":
                return camera, image, gray_img, faces
            for x, y, w, h in faces:
                count = count + 1
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.imwrite(os.path.join(self.pic_repo,"train_image_{0}_{1}.jpg".format(self.owner,count)), gray_img[y:y + h, x:x + w])
            cv2.imshow('image', image)
            cv2.waitKey(0)
        camera.release()
        #cv2.destroyAllWindows()

    def getTrainData(self):
        self.train_X = np.array([])
        self.train_Y = np.array([])
        for root, dir, filename in os.walk(self.pic_repo):
            if not filename.endswith(".jpg"): 
                continue
            index = -1
            for i in range(len(OWNERS)):
                if not OWNERS[i] in filename:
                    continue
                index = i
            if -1 == index:
                continue
            file_path = os.join(root, name)
            face = cv2.imread(file_path)
            np.append(self.train_X, face)
            np.append(self.train_Y, index)

    def train(self):
        self.model = cv2.face.LBPHFaceRecognizer_create()
        self.model.train(np.asarray(self.train_X),np.asarray(self.train_Y))

    def detect(self):
        camera, image, gray_img, faces = catchFacesFromCamera("detect")
        for x, y, w, h in faces:
            pred = self.model.predict(gray_img[y:y + h, x:x + w])
            pred_name = OWNERS[pred]
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image,pred_name,(x,y-20),cv2.FONT_HERSHEY_SIMPLEX,1,255,2)
        cv2.imshow('image', image)
        cv2.waitKey(0)
        camera.release()

if __name__ == '__main__':
    parser = OptionParser() 
    parser.add_option("-c", "--collect", action="store_true", 
                  dest="collect", 
                  default=False, 
                  help="collect face data") 
    parser.add_option("-d", "--detect", action="store_true", 
                  dest="detect", 
                  default=False, 
                  help="detect face") 
    (options, args) = parser.parse_args() 

	assert options.collect and options.detect == False
    if options.collect: 
        detecter = faceDetecter("dorian", "C:\Users\dzi\Pictures")
        detecter.collectFacesFromCamera()
    if options.detect:    
	    detecter = faceDetecter("", "C:\Users\dzi\Pictures")
        detecter.train()
        detecter.detect()

        
