import cv2
from matplotlib.pyplot import contour, gray
import numpy as np
import glob
import random as rng
import skimage
from skimage import measure


img_files = glob.glob('./48369693_artery_enhanced/ori_*.png')
# mask = 0.0

def trackValue(a):
    print(a)

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
cv2.resizeWindow("Parameters", 1024, 1024)
cv2.createTrackbar("slide", "Parameters",0, count, trackValue )
cv2.createTrackbar("Thresh1", "Parameters", 23, 255, trackValue)
cv2.createTrackbar("Thresh2", "Parameters", 20, 255, trackValue)
cv2.createTrackbar("AreaMin", "Parameters", 5000, 3000, trackValue)
cv2.createTrackbar("AreaMax", "Parameters", 5000, 3000, trackValue)


cv2.setTrackbarPos("slide", "Parameters", 13)
cv2.setTrackbarPos("Thresh1", "Parameters", 57)
cv2.setTrackbarPos("Thresh2", "Parameters", 51)
cv2.setTrackbarPos("AreaMin", "Parameters", 0)
cv2.setTrackbarPos("AreaMax", "Parameters", 35)



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
def getContours(threshed, imgCopy):
    contours, hierarchy = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #don't use simple if you wanna get all points  

    for cnt in contours: # ??? ?????? ?????????(??????)??? ??????
        area = cv2.contourArea(cnt) # ?????? ???????????? ????????? ??????

        areaMin = cv2.getTrackbarPos("AreaMin", "Parameters")
        areaMax = cv2.getTrackbarPos("AreaMax", "Parameters")

        if areaMin < area < areaMax: # ????????? ?????? ?????? ???????????? ?????????
            cv2.drawContours(imgCopy, cnt, -1, (255,0, 255), -1) # thickness < 0 ?????? ????????? ?????????

            epsilon = 0.1*cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            ####### ????????? ????????? ??????
            vtc = len(approx)           
            lenn = cv2.arcLength(cnt, True);
            areaDouble = cv2.contourArea(cnt);
            ratioDouble = 4. * np.pi * areaDouble / (lenn * lenn)
            # ????????? ?????? ??????????????? (??????????????? ????????????)
            if (ratioDouble < 1):
                setLabel(imgCopy,"artery",cnt) # ????????? ??? ????????? ???????????? ????????? ??? imgCopy??? ?????? ??????
                return area


while True:  
    slide = cv2.getTrackbarPos("slide", "Parameters") # get the trackbar number 
    img = cv2.imread(img_files[slide], cv2.IMREAD_GRAYSCALE) # CV_LOAD_IMAGE_GRAYSCALE
    ret2, imgThreshed = cv2.threshold(img,0,255,cv2.THRESH_OTSU)

    # ???????????? ?????? ????????? ????????? ?????? : ??????1
    imgCopy = img.copy()
    blank = np.zeros(img.shape, dtype = "uint8") #?????? grayscale?????? shpe[:2]??? ?????? ?????????
    # ?????? ??????????????? blank ?????? ?????? mask??? ????????????, ?????? getContour ?????? ????????? ????????? ?????? ???????????? ??????????????? ????????????, ????????? mask = area??? ??????
    # maskImage= cv2.bitwise_and(img, imgCopy, mask=mask) 
    # maskImage = cv2.copyTo(img, imgCopy, dst)
    
    

    retArea = getContours(imgThreshed, imgCopy)
    retArea = 1 # avoid TypeError: '<' not supported between instances of 'NoneType' and 'int'
    ##### ??????2
    labels = measure.label(imgThreshed, connectivity=2, background=0)
    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue
        # otherwise, construct the label mask and count the
        # number of pixels 
        labelMask = np.zeros(img.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask) # The function returns the number of non-zero elements in src
        # if the number of pixels in the component is sufficiently
        # large, then add it to our mask of "large blobs"
        # if retArea < 100:
        #     mask = cv2.add(blank, labelMask)
    
    for i in range(img.shape[1]):
        mask = 0
        if mask[i] == retArea:
            mask = cv2.add(blank, labelMask)

    imgStack = stackImages(5, ([img, mask]))
    cv2.imshow("Parameters", imgStack)
    if cv2. waitKey(100) & 0xFF == ord('q'):
        break