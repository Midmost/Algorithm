from ast import Try
import cv2
import matplotlib.pyplot
from matplotlib.pyplot import contour
import numpy as np
import glob

img_files = glob.glob('./48369693_artery_enhanced/ori_*.png')


def track(x):
    print(x)


def setLabel(image, str, contour):

    (text_width, text_height), baseline = cv2.getTextSize(str, 
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)

    x,y,width,height = cv2.boundingRect(contour)

    pt_x = x + int((width-text_width)/2)
    pt_y = y + int((height + text_height)/2)

    cv2.rectangle(image, (pt_x, pt_y+baseline), 
        (pt_x+text_width, pt_y-text_height), (200,200,200), cv2.FILLED)
    cv2.putText(image, str, (pt_x, pt_y), cv2.FONT_HERSHEY_SIMPLEX, 
        0.7, (0,0,0), 1, 8)

count = len(img_files)

cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 512, 512)
cv2.createTrackbar("slide", "Parameters",0, count, track )
# cv2.createTrackbar("Thresh1", "Parameters", 23, 255, track)
# cv2.createTrackbar("Thresh2", "Parameters", 20, 255, track)
cv2.createTrackbar("Area", "Parameters", 5000, 3000, track)




#######################################################################
windowWidth = 45
windowLevel = 89

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




def stackImages(scale, allArrays):
    
    np.array(allArrays[0])
    rows = len(allArrays) # 2 : 전체 행렬에서 1차원 행의 갯수만을 알기 위해
    cols = len(allArrays[0]) # 512
    
    rowsAvailable = isinstance(allArrays[0], list) # allArrays[0]가 list맞아? 맞으면 true
    # shape 으로 만들기

    if rowsAvailable is False:        
        for x in range(0, rows):
            if allArrays[x].shape[:2] == allArrays[0].shape[:2]:
                allArrays[x] = cv2.resize(allArrays[x], (0,0), None, scale, scale)
            else:
                allArrays[x] = cv2.resize(allArrays[x], (allArrays[0].shape[1], allArrays[0].shape[0]), None, scale,scale)
           
        hor = np.hstack(allArrays)
        ver = hor
    return ver


# moments
def imageMoments(img, imgContour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #don't use simple if you wanna get all points  
 
    for cnt in contours: # cnt, one of the contour
        area = cv2.contourArea(cnt)
        areaMax = cv2.getTrackbarPos("Area", "Parameters")
        if area < areaMax:       
############## moments calculate   , m00 == cv2.contourArea()
    
            ########################## centroid check
            M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            print('area:', area)
            print("cx:", cx)
            print("cy:", cy)

            cv2.circle(img, (cx, cy), 5, (255, 255, 255), -1)
            cv2.putText(img, "centroid", (cx - 25, cy - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Get the moments
            mu = [None]*len(contours)
            for i in range(len(contours)):
                mu[i] = cv2.moments(contours[i])
            # Get the mass centers
            mc = [None]*len(contours)
            for i in range(len(contours)):
                # add 1e-5 to avoid division by zero
                mc[i] = (mu[i]['m10'] / (mu[i]['m00'] + 1e-5), mu[i]['m01'] / (mu[i]['m00'] + 1e-5))
            # Draw contours

            drawing = np.zeros((imgT.shape[0], imgT.shape[1], 3), dtype=np.uint8)

            for i in range(len(contours)):
                color = (100, 100, 100)
                cv2.drawContours(drawing, contours, i, color, 2)
                cv2.circle(drawing, (int(mc[i][0]), int(mc[i][1])), 4, color, -1)

##########################

    

# input image for contour
def getContours(givemeArray, givemeContour):
    
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #don't use simple if you wanna get all points  
 
    for cnt in contours: # cnt, one of the contour
        area = cv2.contourArea(cnt)
        areaMax = cv2.getTrackbarPos("Area", "Parameters")
        if area < areaMax:   
    
            cv2.drawContours(imgContour, cnt, -1, (255,0, 255), 3)  # contourIdx가 -1 : all the contours are drawn
            epsilon = 0.01*cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
# approxPolyDP
# epsilon – 커질 수록 Point수가 줄어듬
# curve – contours point array 
# closed – 폐곡선 여부
# Returns:	근사치가 적용된 contours point array
   
            vtc = len(approx)
            lenn = cv2.arcLength(cnt, True);
            double_area = cv2.contourArea(cnt)
            try:
                dRatio = 4. * np.pi * double_area / (lenn * lenn)
            except ZeroDivisionError:
                continue

            if (dRatio < 1):
                setLabel(img,"artery",cnt)
            ########


while True:  
    
    ### load
    slide=cv2.getTrackbarPos("slide", "Parameters")
    img = cv2.imread(img_files[slide], cv2.IMREAD_GRAYSCALE)
    
    ### image processing
    matImg_WWL = convertImageWWL(img, [windowLevel, windowWidth])

    ### contouring
    imgContour = matImg_WWL.copy()
    ret, imgT = cv2.threshold(matImg_WWL, 0, 255, cv2.THRESH_OTSU)
    getContours(imgT, imgContour)
    
    ### masked
    # masked()
    
    
    # masked란 함수는 wwl된 이미지의 contour area정보를 받아서 roi area만을 display 해주는 함수야.
    





    imgStack = stackImages(1, [img, matImg_WWL,imgContour])
    cv2.imshow("Parameters", imgStack)
    if cv2. waitKey(100) & 0xFF == ord('q'):
        break
    