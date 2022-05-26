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


def stackImages(scale, imgArray):
    
    np.array(imgArray[0])
    rows = len(imgArray) # 2 : 전체 행렬에서 1차원 행의 갯수만을 알기 위해
    cols = len(imgArray[0]) # 512
    
    rowsAvailable = isinstance(imgArray[0], list) # imgArray[0]가 list맞아? 맞으면 true
    # shape 으로 만들기

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
def getContours(img, imgContour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #don't use simple if you wanna get all points  

    mu = [None]*len(contours)
    for i in range(len(contours)):
        mu[i] = cv2.moments(contours[i])

    for cnt in contours: # cnt, one of the contour
        area = cv2.contourArea(cnt)
        areaMax = cv2.getTrackbarPos("Area", "Parameters")
        if area < areaMax:
            
            ########################## centroid check
            M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            print('area:', area)
            print("cx:", cx)
            print("cy:", cy)

            cv2.circle(img, (cx, cy), 5, (255, 255, 255), -1)
            cv2.putText(img, "centroid", (cx - 25, cy - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
###########################

############## moments calculate   , m00 == cv2.contourArea()

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

# cv2.approx
# epsilon – original cuve와 근사치의 최대거리. 최대거리가 클 수록 더 먼 곳의 Point까지 고려하기 때문에 Point수가 줄어듬
# curve – contours point array 
# closed – 폐곡선 여부
# Returns:	근사치가 적용된 contours point array

      
            
            
            # cv2.drawContours(imgContour, cnt, -1, (255,0, 255), 7)
            # #let's find the corner points
            # # in order to do that, need to get the length
            # epsilon = 0.01*cv2.arcLength(cnt, True)
            # #what type shape?
            # approx = cv2.approxPolyDP(cnt, epsilon, True)

            # ####### 일단 혈관을 동그라미라고 치고
            # # Contour Area = cv2.contourArea(contour) = mmt['m00']
            
            # vtc = len(approx)

            # lenn = cv2.arcLength(cnt, True);
            # double_area = cv2.contourArea(cnt); # contourArea는 input: contour, output: double\\ calculate a contour area
            # try:
            #     double_ratio = 4. * np.pi * double_area / (lenn * lenn)
            # except ZeroDivisionError:
            #     continue

            # if (double_ratio > 0 and double_ratio < 0.8):
            #     setLabel(img,"artery",cnt)
            # ########
            
            # # print(len(approx))
            # x, y, w, h = cv2.boundingRect(approx)
            # cv2.rectangle(imgContour, (x,y), (x+w, y+h), (0,255,0),5)

            # #display value
            # cv2.putText(imgContour, "Points:" + str(len(approx)), (x+w+20, y+20), cv2.FONT_HERSHEY_COMPLEX,.7,(0,255,0),2)
            # cv2.putText(imgContour, "Area:" + str(int(area)), (x+w+20, y+45), cv2.FONT_HERSHEY_COMPLEX,.7,(0,255,0),2)

while True:  
    slide=cv2.getTrackbarPos("slide", "Parameters") # get the trackbar number 
    
    #src를 8bit grayscale로 읽어와야만 에러가 안 뜰거임, type에 0을 쓰면...되나봄...

    img = cv2.imread(img_files[slide], cv2.IMREAD_GRAYSCALE)

        


    ####### 화살표를 누르면서 슬라이드를 움직이기
    ##mac 과 WINDOW는 키값이 다름, WINODW:LEFT-2424832 ,RIGHT-2555904  
    arrow = cv2.waitKey(100) # 1 milliseconds   --> 나의 com 입장에서 1초는 넘 빠름 33초도 넘 빠름 
    if arrow == 2424832: # if pressed left arrow
        while slide > 0 :
            slide -= 1                        
            cv2.setTrackbarPos("slide", 'Parameters', slide)
        continue
    if arrow == 2555904: # if pressed right arrow
        while slide > 0:
            slide += 1
            cv2.setTrackbarPos("slide", 'Parameters', slide)
        continue



    imgContour = img.copy()

    # imgBlur = cv2.GaussianBlur(img, (7,7), 1)
    # imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)

    # threshold1 = cv2.getTrackbarPos("Thresh1", "Parameters") # get the pos of tresh1 in the parameter window
    # threshold2 = cv2.getTrackbarPos("Thresh2", "Parameters")
    
    ret, imgT = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

    #cannyEdgeDetect
    # imgCanny = cv2.Canny(img, threshold1, threshold2)


    # # dilation 위위해  kernel 생성
    # kernel = np.ones((5,5))
    # imgDil = cv2.dilate(imgCanny, kernel, iterations=1)

    # define contour
    getContours(imgT, imgContour)

    # imgT : array([[0,0,0...0],[0,0,0...0]...])
    imgStack = stackImages(2, ([imgT, imgContour]))
    # imgStack = stackImages(2, img, contour)
    cv2.imshow("Parameters", imgStack)
    if cv2. waitKey(100) & 0xFF == ord('q'):
        break
    