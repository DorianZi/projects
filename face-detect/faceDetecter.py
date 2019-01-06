#!/usr/bin/python

import os
import sys
import re
import cv2
#need to install extra module: $pip install opencv-contrib-python
import numpy as np
from optparse import OptionParser

DATA_DIR = "C:\Users\dzi\Pictures"
HAAR_DIR = "C:\Python27\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml"

class faceDetecter():
    def __init__(self,owner,picDir):
        self.pic_repo = picDir
        self.owner = owner
        
    def catchFacesFromCamera(self,mode="collect"):
        camera = cv2.VideoCapture(0)
        haar = cv2.CascadeClassifier(HAAR_DIR)
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
        self.train_X = []
        self.train_Y = []
        self.labelDict = {}
        label_n = 0
        for root, dirs, files in os.walk(self.pic_repo):
            for filename in files:
                try:
                    owner = re.findall(r"train_image_(.*?)_[1-9]\d*\.jpg",filename)[0]
                except:
                    continue
                if not self.labelDict.has_key(owner):
                    label_n = label_n + 1
                    self.labelDict[owner] = label_n 
                file_path = os.path.join(root, filename)
                face = cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)
                if not face.shape == (173,173):
                    face = cv2.resize(face,(173,173))
                self.train_X.append(face)
                self.train_Y.append(label_n)

    def train(self):
        self.model = cv2.face.LBPHFaceRecognizer_create()
        self.model.train(np.asarray(self.train_X),np.asarray(self.train_Y))

    def detect(self):
        camera, image, gray_img, faces = self.catchFacesFromCamera("detect")
        pred_dict = dict([(v,k) for k,v in self.labelDict.items()])
        for x, y, w, h in faces:
            pred = self.model.predict(gray_img[y:y + h, x:x + w])
            pred_name = pred_dict[pred[0]]
            print pred
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
    parser.add_option("-n", "--name", 
                  dest="name",
                  default="", 
                  help="face owner")
    parser.add_option("-d", "--detect", action="store_true", 
                  dest="detect", 
                  default=False, 
                  help="detect face") 
    (options, args) = parser.parse_args() 

    assert (options.collect ^ options.detect)  
    if options.collect:
        assert options.name
        detecter = faceDetecter(options.name, DATA_DIR)
        detecter.collectFacesFromCamera()
    if options.detect:    
        detecter = faceDetecter("", DATA_DIR)
        detecter.getTrainData()
        detecter.train()
        detecter.detect()
        
