from audioop import cross
import cv2 as cv
import numpy as np
import pandas as pd

src = cv.imread('./Lenna.png', cv.COLOR_BGR2RGB)
ddepth = -1
ind = 0
window_a = "a"
window_a_x = "a_x"
window_a_y = "a_y"
window_b = "b"
window_b_x = "b_x"
window_b_y = "b_y"
window_c = "c"
window_c_x = "c_x"
window_c_y = "c_y"
window_c_y_minus = "c_y_minus"

# stackImages



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

b = np.array((
        [3, 0, -3],
        [1, 0, -1],
        [1, 0, -1]), dtype=np.float32)
b_1 = np.array((
        [1, 0, -1],
        [1, 0, -1],
        [3, 0, -3]), dtype=np.float32)
b_2 = np.array((
        [3,0,-3],
        [1,0,-1],
        [3,0,-3]), dtype="float64")
b_3 = np.array((
        [1,0,-1],
        [3,0,-3],
        [1,0,-1]), dtype="float64")



c = np.array(([
        [-3, -1, -1],
        [0, 0, 0],
        [3, 1, 1]]), dtype=np.float32)
c_1 = np.array((
        [-1,-1,-3],
        [0,0,0],
        [1,1,3]), dtype="float64")
c_2 = np.array((
        [-3,-1,-3],
        [0,0,0],
        [3,1,3]), dtype="float64")
c_3 = np.array((
        [-1,-3,-1],
        [0,0,0],
        [1,3,1]), dtype="float64")


d = np.array(([
        [3, 1, 1],
        [0, 0, 0],
        [-3, -1, -1]]), dtype=np.float32)
d_1 = np.array((
        [1,1,3],
        [0,0,0],
        [-1,-1,-3]), dtype="float64")
d_2 = np.array((
        [3,1,3],
        [0,0,0],
        [-3,-1,-3]), dtype="float64")
d_3 = np.array((
        [1,3,1],
        [0,0,0],
        [-1,-3,-1]), dtype="float64")

# Compute the number of rows and columns 
a = cv.filter2D(src, ddepth, a)   
a_1 = cv.filter2D(src, ddepth, a_1) 
a_2 = cv.filter2D(src, ddepth, a_2) 
a_3 = cv.filter2D(src, ddepth, a_3) 
b = cv.filter2D(src, ddepth, b) 
b_1 = cv.filter2D(src, ddepth, b_1)
b_2 = cv.filter2D(src, ddepth, b_2)
b_3 = cv.filter2D(src, ddepth, b_3)
c = cv.filter2D(src, ddepth, c) 
c_1 = cv.filter2D(src, ddepth, c_1)
c_2 = cv.filter2D(src, ddepth, c_2)  
c_3 = cv.filter2D(src, ddepth, c_3)  
d = cv.filter2D(src, ddepth, d) 
d_1 = cv.filter2D(src, ddepth, d_1)
d_2 = cv.filter2D(src, ddepth, d_2)  
d_3 = cv.filter2D(src, ddepth, d_3)  

# kernel_list = [a,a_1,a_2,a_3,a_3, b, b_1,b_2, b_3, c, c_1, c_2, c_3,d,d_1, d_2, d_3]
# for i,v in enumerate(kernel_list):
#     kernel_list[i] = cv.filter2D(src, ddepth, v)
    


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list) # imgArray[0]가 list맞아? 맞으면 true
    # width = imgArray[0][0].shape[1] 이거였는데 두개만 보려고 걍 0으로 써놓음
    width = imgArray[0][0].shape[0]   # a = np.array일 때, a.shape[0]이라는 뜻은, 행의 갯수를 반환하라는 뜻
    height = imgArray[0][0].shape[0]

    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv.resize(imgArray[x][y], (0,0), None, scale, scale)
                else:
                    imgArray[x][y] = cv.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv.cvtColor(imgArray[x][y], cv.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width,3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv.resize(imgArray[x], (0,0), None, scale, scale)
            else:
                imgArray[x] = cv.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale,scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv.cvtColor(imgArray[x], cv.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver    


  
while True:
        
        # a와 a_1 의 차이를 육안으로 확인하기 위해
        # a = cv.cvtColor(a, cv.COLOR_RGB2GRAY)
        # a_1 = cv.cvtColor(a_1, cv.COLOR_RGB2GRAY)
        # diff = cv.absdiff(a, a_1)
        # _, diff = cv.threshold(diff, 2, 255, cv.THRESH_BINARY)
        # cnt, _, stats, _ = cv.connectedComponentsWithStats(diff)
        # for i in range(1, cnt):
        #     x, y, w, h, s = stats[i]
        # res = np.zeros(a.shape, dtype = "uint8")   
        # cv.rectangle(res, (x, y, w, h), (0, 0, 255), 1)

        # 변화구를 준 파일들 모두 모아보기
        imgStack = stackImages(0.5, ([a, a_1, a_2, a_3],[b, b_1, b_2, b_3],[c, c_1, c_2, c_3],[d, d_1, d_2, d_3]))
  
        cv.imshow("Parameters", imgStack)
        # 종료
        quit_button = cv.waitKey(500)
        if quit_button == 27:
            break
        ind += 1
        cv.waitKey()   
cv.destroyAllWindows()
