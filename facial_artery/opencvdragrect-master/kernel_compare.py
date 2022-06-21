import cv2 as cv
import numpy as np
import pandas as pd

src = cv.imread('./Lenna.png', cv.COLOR_BGR2RGB)
ddepth = -1
ind = 0

# kernel series
a = np.array((
        [-3, 0, 3],
        [-1, 0, 1],
        [-1, 0, 1]), dtype="float64")
a_1 = np.array((
        [-1, 0, 1],
        [-1, 0, 1],
        [-3, 0, 3]), dtype="float64")
a_2 = np.array((
        [-3,0,3],
        [-1,0,1],
        [-3,0,3]), dtype="float64")
a_3 = np.array((
        [-1,0,1],
        [-3,0,3],
        [-1,0,1]), dtype="float64")

# Compute the number of rows and columns 
a = cv.filter2D(src, ddepth, a)   
a_1 = cv.filter2D(src, ddepth, a_1) 
a_2 = cv.filter2D(src, ddepth, a_2) 
a_3 = cv.filter2D(src, ddepth, a_3) 

# stackImages
def stackImages(scale, imgAll):
    rows = len(imgAll)
    cols = len(imgAll[0])
    rowsAvailable = isinstance(imgAll[0], list)

    if rowsAvailable is False:
        for x in range(0, rows):
            if imgAll[x].shape[:2] == imgAll[0].shape[:2]:
                imgAll[x] = cv.resize(imgAll[x], (0,0), None, scale, scale)
            else:
                imgAll[x] = cv.resize(imgAll[x], (imgAll[0].shape[1], imgAll[0].shape[0]), None, scale,scale)
            if len(imgAll[x].shape) == 2: imgAll[x] = cv.cvtColor(imgAll[x], cv.COLOR_GRAY2BGR)
        hor = np.hstack(imgAll)
        ver = hor
    return ver
  
while True:

        imgStack_a = stackImages(0.8, [a, a_1, a_2, a_3])
        cv.imshow("Parameters", imgStack_a)
        
        # 종료
        quit_button = cv.waitKey(500)
        if quit_button == 27:
            break
        ind += 1
        cv.waitKey()   
cv.destroyAllWindows()
