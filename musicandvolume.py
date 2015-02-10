import numpy as np
import cv2
import os
import sys
import win32api
import win32con
import time
import subprocess
import pythoncom

task_complete =0
palm_detect=0
face_detect =0
palmx = 0
palmy = 0
facex = 0
facey = 0
facetop=0
facebtm=0
pausecase=0
volumeupcase=0
volumedowncase=0
pauseplaytime=0

def detectproc():
    global palmx
    global palmy
    global facex
    global facey
    global facetop
    global facebot
    global pausecase
    global volumeupcase
    global volumedowncase
    global palm_centroid
    global palm_detect
    global face_detect

    
    if(palmy < facetop and palmy > facebtm):
      if (time.time()>pauseplaytime+2):
          pausecase=1
          print "pause"
          pauseplaytime = time.time()

    if(palmy>facebtm):
      print "volume down"
      print "palm :" +str(palmy)
      print "face :" +str(facetop)
      volumedowncase =1
      
    if(palmy < facetop):
      print"volume up"
      
      volumeupcase=1
    
    pausecase=0
    volumeupcase=0
    volumedowncase=0
    palm_detect=0
    face_detect =0
    return()
  

def getcentroid(x,y,h,w):
  x = x+w/2
  y = y+h/2
  return(x,y)


def movemouse(x,y):
  win32api.SetCursorPos((int(x),int(y)))
  

palm_cascade = cv2.CascadeClassifier('palm.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cam = cv2.VideoCapture(0)
while 1:
  global palm_detect
  global face_detect
  global palmx
  global palmy
  global facex
  global facey
  global facetop
  global facebot
  global plam_cent
  
  _,img = cam.read()
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  palms = palm_cascade.detectMultiScale(gray, 1.3, 5)
  faces = face_cascade.detectMultiScale(gray, 1.3, 5)
  for (ax,ay,aw,ah) in faces:
      cv2.rectangle(img,(ax,ay),(ax+aw,ay+ah),(255,0,0),2)
      #640 by 480
      roi_gray = gray[ay:ay+ah, ax:ax+aw]
      roi_color = img[ay:ay+ah, ax:ax+aw]
      
      face_detect =1 
      facex = ax+aw/2
      facey = ay+ah/2
      facetop = ay
      facebtm = ay+ah
      
    
  for (x,y,w,h) in palms:
      cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
      #640 by 480
      a=(x+(w/2))*6
      b=(y+(h/2))*2
      movemouse(a,b)
      roi_gray = gray[y:y+h, x:x+w]
      roi_color = img[y:y+h, x:x+w]
      palm_detect=1
      palmx = x+w/2
      palmy = y+h/2
      palm_cent = getcentroid(x,y,w,h)
      
  cv2.imshow('img',img)
  detectproc()
  
  cv2.waitKey(27)
cv2.destroyAllWindows()
