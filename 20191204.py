import numpy as numpy
import cv2
import os
import matplotlib.pyplot as plt
import math
import csv
import pandas as pd
from scipy.signal import find_peaks

class VideoCaptureYUV:
    def __init__(self, filename, size):
        self.height, self.width = size
        self.frame_len = self.width * self.height * 3 // 2
        self.f = open(filename, 'rb')
        self.shape = (int(self.height*1.5), self.width)

    def read_raw(self):
        try:
            raw = self.f.read(self.frame_len)
            yuv = numpy.frombuffer(raw, dtype=numpy.uint8)
            yuv = yuv.reshape(self.shape)
        except Exception as e:
            print(str(e))
            return False, None
        return True, yuv

    def read(self):
        ret, yuv = self.read_raw()
        if not ret:
            return ret, yuv
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420, 3)
        return ret, bgr




with open('./test2.csv','rb') as csvfile:
  reader = csv.reader(csvfile)
  column = [row[1] for row in reader]
  column.pop(0)
  column = list(map(int,column))

with open('./test3.csv','rb') as csvfile1:
  reader1 = csv.reader(csvfile1)
  column1 = [row[1] for row in reader1]
  column1.pop(0)
  column1 = list(map(int,column1))




# Open frames in the folder

if __name__ == "__main__":
  filename = "Crew_1280x720_60Hz.yuv"
  size = (720, 1280)
  cap = VideoCaptureYUV(filename, size)
  framey = []
  frameya = []
  frameyam = []
  frameyb = []
  frameybm = []
  frameyc = []
  frameycm = []

  deltay = []
  deltaya = []
  deltayb = []
  deltayc = []

  deltab = []
  deltab2 = []

  s = []

  i = 0
  j = 0

  frameindex = []
  frameindex1 = []
  xa = []
  xb = []
  xc = []
  ymean = []
  previous_gray = cv2.cvtColor(cv2.imread('./allframespng/000000.png'), cv2.COLOR_BGR2GRAY)
  previous_y = numpy.mean(cv2.cvtColor(cv2.imread('./allframespng/000000.png'), cv2.COLOR_BGR2GRAY))
  while 1:
    ret, frame = cap.read()
    if ret:
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      width = 1280
      height = 720
      y1 = numpy.mean(gray)
      framey.append(y1)
      bmax = max(y1, previous_y)
      bi = numpy.power(y1-previous_y,2)
      bii = numpy.sum(bi)
      biii = math.sqrt(bi)
      biiii = abs(y1-previous_y)
      deltay.append(biiii)
      previous_y = y1
      

      for x in range (720):
        for y in range (1280):
                #   a.append(gray[i][j])
                #   b.append(previous_gray[i][j])
          if gray[x][y] > previous_gray[x][y]:
            deltay[x][y] = gray[x][y] - previous_gray[x][y]
            s.append(deltay[x][y])
            print(s)
      print(s)
      mean = numpy.mean(s)
      ymean.append(mean)
      previous_gray = gray
      s = []
      if column[i] == 1:
        xa.append(i)
        deltaya.append(deltay[i])
        frameya.append(framey[i])
        frameyam.append(mean)
      elif column1[i] == 1:    
        xb.append(i)
        deltayb.append(deltay[i])
        frameyb.append(framey[i])
        frameybm.append(mean)
      else:
        xc.append(i)
        deltayc.append(deltay[i])
        frameyc.append(framey[i])
        frameycm.append(mean)
      frameindex1.append(i)

      i = i+1
    else:
      break
  n = len(deltay)
  print(n)
  x = range(0,n)
  for j in range (1,n-1):

    if framey[j]>framey[j-1]+1 and framey[j]>framey[j+1]+1:
      frameindex.append(j)
  print(frameindex)
  print(xa)
  # peaks, _ = find_peaks(framey)
  # print(peaks)
    # else:
        
  # plt.scatter(xa,deltaya,c='red')
  # plt.scatter(xb,deltayb, marker = '^',c='green')
  # plt.scatter(xc,deltayc)
  # plt.plot(frameindex1,deltay,color = 'orange')
  # plt.show()

  plt.scatter(xa,frameyam,c='red')
  plt.scatter(xb,frameybm, marker = '^',c='green')
  plt.scatter(xc,frameycm)
  plt.plot(frameindex1,ymean,color = 'orange')
  plt.show()
