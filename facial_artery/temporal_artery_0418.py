
import cv2
from cv2 import waitKey
from matplotlib.pyplot import contour, gray
import numpy as np
import glob
import random as rng
from skimage import measure
import os
import sys

img_files = glob.glob('./50829509_artery/*.png')

### image
img_height = int(cv2.imread(img_files[0], cv2.IMREAD_UNCHANGED).shape[0])
img_width = int(cv2.imread(img_files[0], cv2.IMREAD_UNCHANGED).shape[1])
img_depth = len(img_files)


### setting
areaMin = 0.2
areaMax = 200

Window_Width = 2
Window_Level = 170


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

# 혈관으로 추정되는 컨투어를 가져오는 함수
def getContours(img_before, img_after):
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
                cv2.drawContours(img_after, cnt, -1, (255,255, 255), -1) # thickness < 0 으로 내부를 그리자
                cv2.fillConvexPoly(img_after, cnt, (255,255,255))
                
                

while True:
    
    for slide in range(img_depth):
        img_ori = cv2.imread(img_files[slide], cv2.IMREAD_GRAYSCALE) # CV_LOAD_IMAGE_GRAYSCALE
        img_WWL = convertImageWWL(img_ori, [Window_Width, Window_Level])  # ww:2, wl:170
        ret2, img_Threshold = cv2.threshold(img_WWL,200,255,cv2.THRESH_BINARY)
        ret3, img_Threshold2 = cv2.threshold(img_Threshold, 200, 255, cv2.THRESH_BINARY)
        

        # 마스크를 씌운 부분만 보이게 하기 : 방법1
        img_masked = np.zeros(img_ori.shape, dtype = "uint8") #이미 grayscale이니 shpe[:2]는 하지 않는다
        getContours(img_Threshold2, img_masked)

        # imgCopy만을 이제 png으로 추출하기 
        img_dst = os.path.join(os.getcwd(), '0403_'+str(slide).zfill(4) + '.png')  
        cv2.imwrite(img_dst, img_masked)

        cv2.imshow("Parameters", img_masked)       
        if cv2. waitKey(100) & 0xFF == ord('q'):
            break



