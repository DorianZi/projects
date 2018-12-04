#!/usr/bin/python

import os
import cv2

class faceDetecter():
    def __init__(self):
        pass
    def getFaceFromCamera(self,outdir):
        camera = cv2.VideoCapture(0)
        haar = cv2.CascadeClassifier('C:\Python27\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
        success, img = camera.read()
        print "success = ",success
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = haar.detectMultiScale(gray_img, 1.3, 5)
        for f_x, f_y, f_w, f_h in faces:
            face = img[f_y:f_y+f_h, f_x:f_x+f_w]
            face = cv2.resize(face, (IMGSIZE, IMGSIZE))
            face = relight(face, random.uniform(0.5, 1.5), random.randint(-50, 50))
            cv2.imwrite(os.path.join(outdir, str(n)+'.jpg'), face)
            cv2.putText(img, 'Dorian', (f_x, f_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            img = cv2.rectangle(img, (f_x, f_y), (f_x + f_w, f_y + f_h), (255, 0, 0), 2)

        cv2.imshow('img', img)
        cv2.waitKey(30)
        camera.release()
        #cv2.destroyAllWindows()

if __name__ == '__main__':
    detecter = faceDetecter()
    detecter.getFaceFromCamera("C:\Users\dzi\Pictures")
    while 1:
        input = input("input a to stop")
        if input == 'a':
            break
        
