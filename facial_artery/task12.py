from turtle import position
from xml.dom import HierarchyRequestErr
import cv2
from cv2 import contourArea
from cv2 import connectedComponentsWithStats
from cv2 import drawContours
from cv2 import connectedComponents
from cv2 import CC_STAT_AREA
from matplotlib.pyplot import contour
import numpy as np
import glob
import os


########## setting for user ##########
path_base = "C:/Users/skia/Algorithm/" 
folder_base = path_base + "facial_artery/"
######################################
idx_start = 0
idx_folder = 1

names_folder = glob.glob(folder_base + '/*')
folder_png = names_folder[idx_folder] + '/'
img_files = glob.glob('./48369693_artery_enhanced/ori_*.png')
count = len(img_files)

def track(x):
    print(x)
    
cv2.namedWindow("task12")
cv2.resizeWindow("task12", 512, 512)
cv2.createTrackbar("slide", "task12",0, count, track )


# for idx_slice in range(idx_start, count):
#     folder_result = folder_png
#     dstImage_WWL = os.path.join(folder_result, 'contour_'+str(idx_slice).zfill(4) + '.png')
#     cv2.imwrite(dstImage_WWL, img_contours)
#     cv2.waitKey(1000)

#######################################################################
windowWidth = 34
windowLevel = 60

# list_windowSetting: [windowLevel, windowWidth]
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
########################################################
def stackImages(scale, imgAll):
    rows = len(imgAll)
    cols = len(imgAll[0])
    rowsAvailable = isinstance(imgAll[0], list)

    if rowsAvailable is False:
        for x in range(0, rows):
            if imgAll[x].shape[:2] == imgAll[0].shape[:2]:
                imgAll[x] = cv2.resize(imgAll[x], (0,0), None, scale, scale)
            else:
                imgAll[x] = cv2.resize(imgAll[x], (imgAll[0].shape[1], imgAll[0].shape[0]), None, scale,scale)
            if len(imgAll[x].shape) == 2: imgAll[x] = cv2.cvtColor(imgAll[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgAll)
        ver = hor
    return ver
   
while True:  
    
    ### load
    slide=cv2.getTrackbarPos("slide", "task12")
    img_ori = cv2.imread(img_files[slide], cv2.IMREAD_GRAYSCALE)
    
    ### image processing
    img_WWL = convertImageWWL(img_ori, [windowLevel, windowWidth])

    ### contouring
    img_contours = np.zeros(img_WWL.shape, np.uint8)
    img_regions = np.zeros(img_contours.shape, np.uint8)
    ret, img_threshold = cv2.threshold(img_WWL, 200, 255, cv2.THRESH_BINARY)
    
    contours, hierarchy = cv2.findContours(img_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    numofcontour, labels, stats, centroids = cv2.connectedComponentsWithStats(img_contours)
    

    for contour in contours:
        area = contourArea(contour)
        if area < 300 and area >10: 
            small_area = cv2.fillConvexPoly(img_contours,contour,color=(255,0,0)) # LINE_AA가 안 먹고 왜 이렇게 했을 때 속이 채워지게 되는거지?
            regions, hierarchy_region = cv2.findContours(small_area, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE ) #칠해진 각 각의 작은 영역들을 다시 컨투어로
            dstImage_WWL = os.path.join(folder_base, 'contour_'+str(slide).zfill(4) + '.png')
            slide +=1
            cv2.imwrite(dstImage_WWL, img_contours)
            
# 여기까지 한 다음 img_contours 만을 뽑아서 imageJ를 볼 수도 있고
# 아니면 두개골을 기준으로 두개골 안 쪽의 혈관은 모두 버려버리는 것
# 두개골 안 쪽의 혈관을 모두 버리는 접근 방법은 다양한데, contour를 따놓으 다음 ccwithstats를 이용해서 area가 가장 큰 면적일 때 안을 비우는 것
# threshold를 이용해서 subtract를 하는 것
                
    imgStack = stackImages(1, [img_ori, img_WWL, img_threshold, img_contours, img_regions])
    cv2.imshow("task12", imgStack)
    if cv2. waitKey(100) & 0xFF == ord('q'):
        break
