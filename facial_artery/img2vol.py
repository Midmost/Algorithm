# image(png) → *.vol



import sys
import os
import glob
import cv2
import numpy as np

from withMarsprocessor import voltool



########## parameter for user ##########
idx_folder = 0

nameVol_mask = 'vol_mask'
########################################

# ########## setting for user ##########
# path_base = "C:/dsyoon/Python/Study/skia/data/" 
# folder_base = path_base + "mask2vol/"
# ######################################

folder_base = 'C:/Users/skia/Algorithm/facial_artery/'

## main
### files
path_folders = glob.glob(folder_base + '/*')

folder_img = path_folders[idx_folder] + '/'
img_files = glob.glob(folder_img + '/*.png')
if not img_files:
    sys.exit()


### image
img_height = int(cv2.imread(img_files[0], cv2.IMREAD_UNCHANGED).shape[0])
img_width = int(cv2.imread(img_files[0], cv2.IMREAD_UNCHANGED).shape[1])
img_depth = len(img_files)


## action
listImg_mask = list()

idx_start = 0
for idx_slice in range(idx_start, len(img_files)):
    print("index image = ", idx_slice)

    # load
    matImg_ori = cv2.imread(img_files[idx_slice], cv2.IMREAD_UNCHANGED)



    ## check
    ### display 
    cv2.imshow('original', matImg_ori)
    cv2.waitKey(50)


    ### save        
    #### vol
    matImg_ori[matImg_ori > 0] = 1 # *.vol에서는 '1'을 mask로 인식. 
    listImg_mask.append(matImg_ori)
arrImg_mask = np.array(listImg_mask, dtype=np.int16) # *.vol 자료형

#### vol
folder_save_vol = folder_img
file_save_vol = folder_save_vol + nameVol_mask + '.vol'

vol_default = voltool.SkiaVolume(glob.glob(folder_base + '/*.vol')[0])
#vol_default.res = np.array([vol_default.res[0], vol_default.res[1], vol_default.res[2]*2], dtype=np.float32) # CT 해상도가 다를 때도 있음
vol_default.dim = np.array([img_height, img_width, img_depth])
vol_default.data = arrImg_mask
vol_default.minmax = b'\x00\x00\x01\x00' # min: 0, max: 1 
vol_default.writeVolume(file_save_vol)