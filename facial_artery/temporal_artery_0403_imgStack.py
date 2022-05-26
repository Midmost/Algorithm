
import cv2
from cv2 import waitKey
from cv2 import imshow
from matplotlib.pyplot import contour, gray
import numpy as np
import glob
import random as rng
from skimage.filters import frangi, hessian
import os
import sys

img_files = glob.glob('./50829509_artery/*.png')
count = len(img_files)
cranium = count * (1/3)    #190


### setting
areaMin = 1
areaMax = 200

def trackValue(a):
    print(a)

cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 1024, 1024)
cv2.createTrackbar("Slide", "Parameters",0, count, trackValue)
cv2.createTrackbar("Window_Width", "Parameters", 2, 4094, trackValue)
cv2.createTrackbar("Window_Level", "Parameters", 0, 4095, trackValue) # -1024, 3071 이지만 trackbar가 항상 0부터 시작하므로

#선택한 컨투어에 라벨을 달아주는 함수
def setLabel(image, str, contour):

    (text_width, text_height), baseline = cv2.getTextSize(str, 
        cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)

    x,y,width,height = cv2.boundingRect(contour)

    pt_x = x + int((width-text_width)/2)
    pt_y = y + int((height + text_height)/2)

    cv2.rectangle(image, (pt_x, pt_y+baseline), 
        (pt_x+text_width, pt_y-text_height), (100,300,100), cv2.FILLED)
    cv2.putText(image, str, (pt_x, pt_y), cv2.FONT_HERSHEY_SIMPLEX, 
        0.3, (0,0,0), 1, 5)

# 대수씌 함수
def convertImageWWL(matImg_source8bit, list_windowSetting):

    WindowLevel = list_windowSetting[0]
    WindowWidth = list_windowSetting[1]
    WindowLow = round(WindowLevel - WindowWidth/2)
    WindowHigh = round(WindowLevel + WindowWidth/2)
   
    matImg_target8bit = matImg_source8bit.copy()
    matImg_target8bit[matImg_target8bit < WindowLow] = WindowLow
    matImg_target8bit[matImg_target8bit > WindowHigh] = WindowHigh
    matImg_target8bit = cv2.normalize(matImg_target8bit, None, 0, 255, cv2.NORM_MINMAX)

    return matImg_target8bit

# 윈도우들을 순차적으로 모아서 볼 수 있게 해주는 함수
def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[0]
    height = imgArray[0][0].shape[0]

    if rowsAvailable is False:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0,0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale,scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver

# 혈관으로 추정되는 컨투어를 가져오는 함수
def getContours(img_before, img_after, img_labeled):
    contours, hierarchy = cv2.findContours(img_before, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #don't use simple if you wanna get all points  
    for cnt in contours: # 각 각의 컨투어(영역)을 구함
        area = cv2.contourArea(cnt) # 개별 컨투어의 면적을 구함

        if areaMin < area < areaMax: # 면적이 내가 원한 사이즈에 맞으면           
            epsilon = 0.1*cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            ####### 동그란 혈관의 경우
            vtc = len(approx)           
            lenn = cv2.arcLength(cnt, True);
            areaDouble = cv2.contourArea(cnt);
            ratioDouble = 4. * np.pi * areaDouble / (lenn * lenn)
            # 혈관의 각이 둥그렇다면 (동그라미에 가깝다면)
            if (ratioDouble < 1):
                # 새로 만든 배열에 혈관만 속을 채워 그려줘
                setLabel(img_labeled,"artery",cnt) 
                cv2.drawContours(img_after, cnt, -1, (255,255, 255), -1) # thickness < 0 으로 내부를 그리자
                cv2.fillConvexPoly(img_after, cnt, (255,255,255))



                
while True:
    slide = cv2.getTrackbarPos("Slide", "Parameters")
    slide_min = cv2.getTrackbarPos("Window_Width", "Parameters")
    slide_max = cv2.getTrackbarPos("Window_Level", "Parameters")
    
    img_ori = cv2.imread(img_files[slide], cv2.IMREAD_GRAYSCALE) # CV_LOAD_IMAGE_GRAYSCALE
    img_WWL = convertImageWWL(img_ori, [slide_min, slide_max])
    ret2, img_Threshold = cv2.threshold(img_WWL, 200, 255, cv2.THRESH_BINARY)


    # 마스크를 씌운 부분만 보이게 하기 : 방법1
    img_labeled = img_ori.copy()
    img_masked = np.zeros(img_ori.shape, dtype = "uint8") #이미 grayscale이니 shpe[:2]는 하지 않는다
    getContours(img_Threshold, img_masked, img_labeled)
  
    # 한 화면에 순차적으로 이미지 처리 과정 보여주기
    imgStack = stackImages(0.8, ([img_labeled, img_Threshold, img_masked]))
    # cv2.imshow("frangi",frangi(img_Threshold))
    cv2.imshow("Parameters", imgStack)
    if cv2. waitKey(100) & 0xFF == ord('q'):
        break



