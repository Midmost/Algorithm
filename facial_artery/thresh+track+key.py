# https://manivannan-ai.medium.com/set-trackbar-on-image-using-opencv-python-58c57fbee1ee

from curses import window
import os
import cv2
import numpy as np
import glob

def nothing(x):
  pass

img_files = glob.glob('./48369693_artery_enhanced /WWL_*.png')
cv2.namedWindow('image', cv2.WINDOW_NORMAL)

#wnd size
# cv2.setWindowProperty 함수를 사용하여 속성 변경
# cv2. WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN을 이용하여 전체화면 속성으로 변경
# cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cnt = len(img_files)
idx = 0

####
cv2.createTrackbar("Next_image", "image", 0, cnt, nothing )
cv2.createTrackbar("min", "image",0,255,nothing)
cv2.createTrackbar("max", "image",0,255,nothing)

## 처음에 안 보여서 
cv2.setTrackbarPos("max", "image", 200)

while True:
  ########## ** trackbar idx
    slide=cv2.getTrackbarPos("Next_image", "image") # get the trackbar number   
    
    ####### 화살표를 누르면서 슬라이드를 움직이기
    arrow = cv2.waitKey(100) # 1 milliseconds   --> 나의 com 입장에서 1초는 넘 빠름 33초도 넘 빠름 
    if arrow == 2: # if pressed left arrow
      slide -= 1                        #key를 누른 동안에만 이미지가 바뀜, 내가 원하는 건 누른 뒤에 trackbar 자체의 숫자도 바뀌는 거임
      cv2.setTrackbarPos("Next_image", 'image', slide)

    if arrow == 3: # if pressed right arrow
      slide += 1
      cv2.setTrackbarPos("Next_image", 'image', slide)


    ##### Exception for arrow key
    # 만약에 키값이 가장 끝에 다다르면 프로그램이 자동종료가 되어버림 -> 자동종료하지 말고 마지막장이란 프린트보여주기
    
      
    ###### 
    img = cv2.imread(img_files[slide])   

    #######  thresh
    hul=cv2.getTrackbarPos("min", "image")
    huh=cv2.getTrackbarPos("max", "image")   
    ret,thresh1 = cv2.threshold(img,hul,huh,cv2.THRESH_BINARY)
  
    if img is None:
        print('Image load failed')
        break

    # cv2.imshow('image', img)
    cv2.imshow("image",thresh1)

    if slide >= cnt:
        break

    k = cv2.waitKey(100) & 0xFF
    if k == ord('q'): # if q pressed, quit the program
      mode = not mode
    elif k == 27: # esc, quit the program
      break
        
cv2.destroyAllWindows()

