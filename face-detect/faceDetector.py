#!/usr/bin/python

import os
import sys
import re
import cv2
from sklearn.svm import SVC
#need to install extra module: $pip install opencv-contrib-python
import numpy as np
import logging
from optparse import OptionParser

DATA_DIR = "C:\Users\dzi\Pictures"
HAAR_DIR = "C:\Python27\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml"
PROTOTXT_FILE = "deploy.prototxt"
MODEL_FILE = "res10_300x300_ssd_iter_140000.caffemodel"

def ERR_EXIT(err=""):
    if err: 
        logging.error(err)
    sys.exit(-1)

class faceDetecter():
    def __init__(self,owner,picDir):
        self.pic_repo = picDir
        self.owner = owner
        modelDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "model")
        prototxtPath = os.path.join(modelDir, PROTOTXT_FILE)
        modePath = os.path.join(modelDir, MODEL_FILE)
        self.caffe_detector = cv2.dnn.readNetFromCaffe(prototxtPath, modePath)

    def catchFacesFromCamera(self,camera,mode="collect"):
        count = 0
        for n in range(30):
            success, image = camera.read()
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            (h, w) = image.shape[:2]
            imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), 
                                               (104.0, 177.0, 123.0), swapRB=False, crop=False)
            self.caffe_detector.setInput(imageBlob)
            detections = self.caffe_detector.forward()
            faceBoxList = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence >  0.5:
                    faceBox = (detections[0, 0, i, 3:7] * np.array([w, h, w, h])).astype("int")
                    (startX, startY, endX, endY) = faceBox
                    faceBoxList.append(faceBox)
            if mode == "detect":
                return image, gray_img, faceBoxList
            for startX, startY, endX, endY in faceBoxList:
                count = count + 1
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.imwrite(os.path.join(self.pic_repo,"train_image_{0}_{1}.jpg".format(self.owner,count)), cv2.resize(gray_img[startY:endY, startX:endX],(300,300)))
            cv2.imshow('image', image)
            cv2.waitKey(0)
        #cv2.destroyAllWindows()

    def catchFacesFromPicture(self,pic):
        pic_path = pic
        if not ("/" in pic or "\\" in pic):
            pic_path = os.path.join(self.pic_repo,pic)
        image = cv2.imread(pic_path)
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        (h, w) = image.shape[:2]
        imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), 
                                           (104.0, 177.0, 123.0), swapRB=False, crop=False)
        self.caffe_detector.setInput(imageBlob)
        detections = self.caffe_detector.forward()
        faceBoxList = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence >  0.5:
                faceBox = (detections[0, 0, i, 3:7] * np.array([w, h, w, h])).astype("int")
                faceBoxList.append(faceBox)
        return image,gray_img,faceBoxList

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
                face_img = cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)
                face_img = cv2.resize(face_img,(300,300))
                self.train_X.append(face_img)
                self.train_Y.append(label_n)
        self.train_X, self.train_Y = np.asarray(self.train_X), np.asarray(self.train_Y)

    def train(self, algorithm="SVM"):
        self.algorithm = algorithm
        if self.algorithm == "SVM":
            self.model = SVC(C=1.0, kernel="linear", probability=True)
            self.model.fit(self.train_X.reshape(self.train_X.shape[0],-1), self.train_Y)
        if self.algorithm == "EigenFace":
            self.model = cv2.face.EigenFaceRecognizer_create()
            self.model.train(self.train_X, self.train_Y)


    def detect(self,source="",camera="",pic="",video=False):
        if source == "camera":
            image, gray_img, faces = self.catchFacesFromCamera(camera,"detect")
        elif source == "picture":
            image, gray_img, faces = self.catchFacesFromPicture(pic)
        pred_dict = dict([(v,k) for k,v in self.labelDict.items()])
        for startX, startY, endX, endY in faces:
            target = cv2.resize(gray_img[startY:endY, startX:endX],(300,300))
            target = target.reshape(1,-1) if self.algorithm == "SVM" else target
            pred = self.model.predict(target)
            if self.algorithm == "SVM": 
                pred_proba = self.model.predict_proba(target)[0][pred[0]-1]
            pred_name = pred_dict[pred[0]]
            show_text = (pred_name + " %.2f%%" % (100*float(pred_proba))) if self.algorithm == "SVM" else pred_name
            print (pred,pred_proba) if self.algorithm == "SVM" else pred
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(image,show_text,(startX,startY-20),cv2.FONT_HERSHEY_SIMPLEX,1,255,2)
        cv2.imshow('image', image)
        if video:
            if not cv2.waitKey(10) == -1:
                return False
        else:
            cv2.waitKey(0)
        return True

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
    parser.add_option("-p", "--pic",
                  dest="pic",
                  default="",
                  help="picture input") 
    parser.add_option("-v", "--video", action="store_true",
                  dest="video",
                  default=False,
                  help="detect in video stream") 
    (options, args) = parser.parse_args() 
    if not (options.collect ^ options.detect):
        ERR_EXIT("require either --collect/-c or --detect/-d specified !")
    if options.collect:
        if not options.name:
            ERR_EXIT("require --name/-n specified!")
        detecter = faceDetecter(options.name, DATA_DIR)
        cmr = cv2.VideoCapture(0)
        detecter.catchFacesFromCamera(cmr)
        cmr.release()
    if options.detect:
        detecter = faceDetecter("", DATA_DIR)
        detecter.getTrainData()
        detecter.train()
        if options.pic:
            detecter.detect(source="picture",pic=options.pic)
        else:
            cmr = cv2.VideoCapture(0)
            if options.video:
                while 1:
                    if not detecter.detect(source="camera",camera=cmr,video=True):
                        break
            else:
                detecter.detect(source="camera",camera=cmr)
            cmr.release()
