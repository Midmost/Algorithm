import cv2

k = cv2.waitKey(100) 

img = cv2.imread('C:/Users/skia/Algorithm/facial_artery/dof.png')

while(1):
    cv2.imshow('img',img)
    if k == 27:
        break
    elif k == -1:
        continue
    else:
        print(k)
