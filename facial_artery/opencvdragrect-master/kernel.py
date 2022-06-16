from audioop import cross
import cv2 as cv
import numpy as np
import pandas as pd

src = cv.imread('./Lenna.png', cv.COLOR_BGR2RGB)
ddepth = -1
ind = 0
window_name = 'filter2D'

# Set recursion limit
import sys
sys.setrecursionlimit(10 ** 9)
import selectinwindow


# Initialize the  drag object
wName = "select region"
imageWidth = src.shape[0] # 이게 왜 안 먹는 걸까?
imageHeight = src.shape[1]

# kernel series
def kernels():
        a = np.array(([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]]), dtype=np.float32)
        b = np.array((
        [0,0,1],
        [0,0,1,],
        [0,0,1]), dtype="float64")

        c = np.array((
                [-1,0,1],
                [-2,0,2],
                [-1,0,1]), dtype="float64")
        d = np.array((
                [-2,0,2],
                [-1,0,1],
                [-1,0,1]), dtype="float64")
        e = np.array((
                [-1,0,1],
                [-1,0,1],
                [-2,0,2]), dtype="float64")
        e = np.array((
                [-1,0,1],
                [-2,0,2],
                [-3,0,3]), dtype="float64")

a = np.array(([
        [-3, 0, 3],
        [-1, 0, 1],
        [-1, 0, 1]]), dtype=np.float32)

b = np.array(([
        [-1, 0, 1],
        [-1, 0, 1],
        [-3, 0, 3]]), dtype=np.float32)

c = np.array((
        [-3,0,3],
        [-1,0,1],
        [-3,0,3]), dtype="float64")

d = np.array((
        [-1,0,1],
        [-3,0,3],
        [-1,0,1]), dtype="float64")



kernel_size = 3
kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
kernel /= (kernel_size * kernel_size)

# Compute the number of rows and columns 
dst = cv.filter2D(src, ddepth, d)   
rectI = selectinwindow.DragRectangle(dst, wName, 1000, 1000) # imageHeight imageWidth 안 먹음 그래서 임의로 1000으로 설정
cv.namedWindow(rectI.wname)
cv.setMouseCallback(rectI.wname, selectinwindow.dragrect, rectI)

while True:

        # Define the drag object
        x = rectI.outRect.x
        y = rectI.outRect.y
        w = rectI.outRect.w
        h = rectI.outRect.h     
        
        cropped = rectI.image[y:y+h, x:x+w]
        print(cropped)
        
        cv.imshow(wName, rectI.image)
        
        # if returnflag is True, break from the loop
        if rectI.returnflag:
                break
        
        # 종료
        quit_button = cv.waitKey(500)
        if quit_button == 27:
            break
        ind += 1
        cv.waitKey()   
     
newarr = cropped.reshape(cropped.shape[0], (cropped.shape[1]*cropped.shape[2]))     
print(newarr)   
df = pd.DataFrame(newarr, dtype=np.float32)
df.to_csv('./results.csv', index=False, encoding='utf-8')
         
print(str(rectI.outRect.x) + ',' + str(rectI.outRect.y) + ',' + \
        str(rectI.outRect.w) + ',' + str(rectI.outRect.h))  

cv.destroyAllWindows()
