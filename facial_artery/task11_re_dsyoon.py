import cv2
from cv2 import findContours
from matplotlib.pyplot import contour
import numpy as np
import glob
import skimage
from skimage import measure

img_files = glob.glob('./48369693_artery_enhanced/ori_*.png')
count = len(img_files)

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

img_ori = cv2.imread(img_files[0], cv2.IMREAD_GRAYSCALE)

### image processing
img_WWL = convertImageWWL(img_ori, [windowLevel, windowWidth])

cv2.imshow("ori", img_ori)
cv2.imshow("WWL", img_WWL)



img_contours = np.zeros(img_WWL.shape, np.uint8)
contours, hierarchy = cv2.findContours(img_WWL, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 

for contour in contours:
    cv2.fillPoly(img_contours, contour, 255)
    cv2.imshow("contours", img_contours)
    
    
    cv2.waitKey()


