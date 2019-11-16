import numpy as numpy
import cv2
import os
import matplotlib.pyplot as plt
import csv
import glob
import math
from PIL import Image
import sys
import pandas as pd

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

with open('./test.csv','r') as csvfile:
  reader = csv.reader(csvfile)
  column = [row[1] for row in reader]
  column.pop(0)
  column = list(map(int,column))


deltay = numpy.zeros((720,1280))
deltau = numpy.zeros((720,1280))
deltav = numpy.zeros((720,1280))
ye = numpy.zeros((720,1280))
ue = numpy.zeros((720,1280))
ve = numpy.zeros((720,1280))


k = 0
# cap = cv2.VideoCapture("test.mp4")
# ret, frame = cap.read()
if __name__ == "__main__":
    #filename = "data/20171214180916RGB.yuv"
    filename = "Crew_1280x720_60Hz.yuv"
    size = (720, 1280)
    cap = VideoCaptureYUV(filename, size)
    ret, frame = cap.read()

    fps = 5   
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')  
    videoWriter = cv2.VideoWriter('./testcomp.mp4', fourcc, fps, (1280,720))
    count = 0
    videoWriter.write(frame)

   
    # Open frames in the folder
    for n in range(10):
        prev_img = frame.copy()
        a = []
        b = []
        c = []
        ret, frame = cap.read()
        #   cv2.imshow("frame", frame)
        #   if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        img = frame
        #   gray_levels = 256
        #   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        gray = yuv[...,0]
        U = yuv[...,1]  #0.492 * (img[...,0] - gray)
        V = yuv[...,2]  #0.877 * (img[...,2] - gray)

        prev_yuv = cv2.cvtColor(prev_img, cv2.COLOR_BGR2YUV)
        prev_gray = prev_yuv[...,0]
        prev_U = prev_yuv[...,1]
        prev_V = prev_yuv[...,2]
        for i in range (720):
            for j in range (1280):
            #   a.append(gray[i][j])
            #   b.append(previous_gray[i][j])
                if gray[i][j] > prev_gray[i][j]:
                    deltay[i][j] = gray[i][j] - prev_gray[i][j]
                    ye[i][j] = 1
                else:
                    deltay[i][j] = prev_gray[i][j] - gray[i][j]
                    ye[i][j] = -1
                if U[i][j] > prev_U[i][j]:
                    deltau[i][j] = U[i][j] - prev_U[i][j]
                    ue[i][j] = 1
                else:
                    deltau[i][j] = prev_U[i][j] - U[i][j]
                    ue[i][j] = -1
                if V[i][j] > prev_V[i][j]:
                    deltav[i][j] = V[i][j] - prev_V[i][j]
                    ve[i][j] = 1
                else:
                    deltav[i][j] = prev_V[i][j] - V[i][j]
                    ve[i][j] = -1
                #   c.append(ye[i][j]*deltay[i][j])
                


        # print(hist)
        # im = frame
        # pix = im.load()
        width = 1280
        height = 720
        pixelindexy = numpy.zeros((720,1280))
        pixelindexu = numpy.zeros((720,1280))
        pixelindexv = numpy.zeros((720,1280))
        if column[n+1] == 1:
            # print(prev_img)
            for y in range (720):
                for x in range (1280):
                    # pixelindexy[y][x] = ye[y][x]*deltay[y][x]
                    # pixelindexu[y][x] = ue[y][x]*deltau[y][x]
                    # pixelindexv[y][x] = ve[y][x]*deltav[y][x]
                    if deltay[y][x] > 5 and ye[y][x] > 0 and ve[y][x] < 0 and ue[y][x] > 0:  #:
                        # print(y,x)          
                        img[y,x,2] = img[y,x,2] - ye[y][x]*deltay[y][x] - ve[y][x]*1.14*deltav[y][x]
                        img[y,x,1] = img[y,x,1] - ye[y][x]*deltay[y][x] + ue[y][x]*0.395*deltau[y][x] + ve[y][x]*0.581*deltav[y][x]
                        img[y,x,0] = img[y,x,0] - ye[y][x]*deltay[y][x] - ue[y][x]*2.033*deltau[y][x]
                    # elif deltay[y][x] > 5 and ye[y][x] > 0 and deltav[y][x] > 2 and ve[y][x] > 0 and deltau[y][x] > 4 and ue[y][x] > 0:  #:
                    #     # print(y,x)          
                    #     img[y,x,2] = img[y,x,2] - ye[y][x]*deltay[y][x] - ve[y][x]*1.14*deltav[y][x]
                    #     img[y,x,1] = img[y,x,1] - ye[y][x]*deltay[y][x] + ue[y][x]*0.395*deltau[y][x] + ve[y][x]*0.581*deltav[y][x]
                    #     img[y,x,0] = img[y,x,0] - ye[y][x]*deltay[y][x] - ue[y][x]*2.033*deltau[y][x]
                

            print(k+1,'flash')
            cv2.imwrite('%.3d-comp.png'%count,img)
            videoWriter.write(img)
            # plt.figure('deltay')
            # plt.imshow(numpy.flipud(pixelindexy),interpolation='nearest',cmap='bone',origin='lower')
            # plt.colorbar()
            # plt.xticks(())
            # plt.yticks(())
            # plt.figure('deltau')
            # plt.imshow(numpy.flipud(pixelindexu),interpolation='nearest',cmap='bone',origin='lower')
            # plt.colorbar()
            # plt.xticks(())
            # plt.yticks(())
            # plt.figure('deltav')
            # plt.imshow(numpy.flipud(pixelindexv),interpolation='nearest',cmap='bone',origin='lower')
            # plt.colorbar()
            # plt.xticks(())
            # plt.yticks(())
            # plt.show()
        else:
            print(k+1,'no flash')
            videoWriter.write(img)

            

            
            # plt.imshow(adjustedpixel, interpolation='nearest')
            # plt.show()
            # im.show()


        count += 1
        k += 1
    videoWriter.release()
