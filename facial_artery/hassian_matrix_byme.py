from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from PIL import Image
import numpy as np
import cv2

# image = camera()
img_ori = np.asarray(Image.open('dof.png'))
#imread 대신에 ASARRAY를 썼음 img_ori = cv2.imread(img_files[slide], cv2.IMREAD_GRAYSCALE)
# np.asarray 결과물이랑 cv2의 image라는 이름의 결과물이랑...다른건가? 둘다 그냥 배열일 줄 알았는데...? 

cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 1024, 1024)

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

# 빈 이미지를 만들고 거기에 original image copy 한 거에 frangi
# img_frangi = frangi(img_ori.copy())

hxx, hxy, hyy = hessian_matrix(img_ori, sigma = 3.0)

i1, i2 = hessian_matrix_eigvals([hxx, hxy, hyy])


while True:
    imgStack = stackImages(0.8, ([i1, i2]))
    
    cv2.imshow("Parameters", imgStack)
    if cv2. waitKey(100) & 0xFF == ord('q'):
        break

import logging 
logging.basicConfig(filename='<filename>.log', level=logging.INFO)