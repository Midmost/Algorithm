```python3
import cv2
from cv2 import Mat
import numpy as np


src = cv2.imread('ori.png', cv2.IMREAD_GRAYSCALE)

dx = cv2.Sobel(src, -1, 1, 0, delta=128) # delta값을 지정 안 하면 - 부분이 미분시 0으로 나옴
dy = cv2.Sobel(src, -1, 0, 1, delta=128) # ddepth = -1 이면 원본과 같은 데이터타입
both = cv2.Sobel(src,-1, 1,1 )

print(f"src: {src}")
print(f"dy: {dy}")
print(f"dx: {dx}")
print(f"bpth: {both}")


cv2.imshow('src', src)
# cv2.imwrite('src.png', src)
cv2.imshow('dy', dy)
# cv2.imwrite('dy.png', dy)
cv2.imshow('dx', dx)
# cv2.imwrite('dx.png', dx)
cv2.imshow('both', both)

cv2.waitKey()

cv2.destroyAllWindows()

```

이미지 추가는...귀찮다...bitbucket에 있으니까...그냥 추가 안 
