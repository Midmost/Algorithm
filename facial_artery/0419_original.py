

import cv2
from matplotlib.pyplot import contour, gray
import numpy as np
import glob
import random as rng

# 원본 파일 경로, 만약 다른 컴퓨터에서 연다면 아래 경로를 변경해줘야 함. 
img_files = glob.glob('./48369693_artery_enhanced/ori_*.png')

# 트랙바 값이 변경되는 것을 보여주기 위한 함수, 여기서는 몇 번째 슬라이스인지를 알려준다. 
def trackValue(a):
    print(a)

# 매번 puttext를 통해 라벨링해주기 귀찮아서 복붙하여 사용할 함수
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

#count는 img_depth를 의미함
count = len(img_files)

# 화면상에 나올 UI
cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 1024, 1024)
cv2.createTrackbar("slide", "Parameters",0, count, trackValue )

# 한 화면에 윈도우를 모아서 보여주기 위한 함수
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

# input image for contour
def getContours(img_all, imgCopy):
    contours, hierarchy = cv2.findContours(img_all, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #don't use simple if you wanna get all points  
    for cnt in contours: # 각 각의 컨투어(영역)을 구함
        area = cv2.contourArea(cnt) # 개별 컨투어의 면적을 구함

        if 1 < area < 300: # 면적이 내가 원한 사이즈에 맞으면           
            epsilon = 0.1*cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            ####### 동그란 혈관의 경우
            vtc = len(approx)           
            lenn = cv2.arcLength(cnt, True);
            areaDouble = cv2.contourArea(cnt);
            ratioDouble = 4. * np.pi * areaDouble / (lenn * lenn)
            # 혈관의 각이 둥그렇다면 (동그라미에 가깝다면)
            if (ratioDouble < 1):
                setLabel(imgCopy,"artery",cnt)
                cv2.drawContours(imgCopy, cnt, -1, (255,255, 255), -1) # thickness < 0 으로 내부를 그리자

# main 함수 : wwl먹음 이미지, 거기에 threshold를 먹인 이미지, 원본에 라벨링이 붙은 이미지 이렇게 3개가 순서대로 보일 수 있도록 하였다. 
while True:  
    slide = cv2.getTrackbarPos("slide", "Parameters") # get the trackbar number 
    img = cv2.imread(img_files[slide], cv2.IMREAD_GRAYSCALE) # CV_LOAD_IMAGE_GRAYSCALE
    img_WWL = convertImageWWL(img, [34, 120])
    ret2, imgThreshed = cv2.threshold(img_WWL,200,255,cv2.THRESH_BINARY)

    imgCopy = img.copy()
    getContours(imgThreshed, imgCopy)

    imgStack = stackImages(2, ([img_WWL, imgThreshed, imgCopy]))
    cv2.imshow("Parameters", imgStack)
    if cv2. waitKey(100) & 0xFF == ord('q'):
        break
