# ==> 폴더명과 같은 .nrrd 파일이 존재하지 않는지 알려준다.

import os
import glob

def get_dir_name(dir_path):
    # folder의 path를 받으면 folder_name을 str으로 출력
    # 입력: "C:/Users/skia/Downloads/sample/test2"
    # 출력: "test2"
    # folder_name = path.split('/')[-1] # 이런식으로 뜰 수 있기 때문에 'alkee-testing\\29539554-75785777-7-CAC' 
    folder_name = dir_path.split(os.path.sep)[-1]
    return folder_name


def exist(nrrd_path):
    # nrrd_path가 있는가 없는가
    # 입력: "C:/Users/skia/Downloads/sample/test2/test2.nrrd"
    # 출력: "C:/Users/skia/Downloads/sample/test2/test2.nrrd" 파일이 존재한다면 True 나머지는 모두 False
    return os.path.isfile(nrrd_path)

def contains_same_dir_name_nnrd(full_dir_path):
    # 폴더와 같은 이름의 nrrd파일을 갖고있는가
    # 입력: "C:/Users/skia/Downloads/sample/test2"
    folder_name = get_dir_name(full_dir_path)
    # 출력: "C:/Users/skia/Downloads/sample/test2/test2.nrrd" 파일이 있으면 True 나머지는 모두 False ,. 같은 거를 생각하면 경우의수가 더 적고 단순하다아아ㅏㅏ
    nrrd_name =  f"{folder_name}.nrrd"
    # nrrd_path = full_dir_path + "/" + nrrd_name    # 이것도 split과 마찬가지로 슬래시 문제가 생길 수 있기에
    nrrd_path = os.path.join(full_dir_path, nrrd_name)
    if exist(nrrd_path):
       return True 
    return False

def get_all_child_path(path_root):
    # 부모 dir 의 자식 dir들을 list로 반환하여라
    # 입력 : "C:/Users/skia/Downloads/sample"
    # 출력 : ["C:/Users/skia/Downloads/sample/test1", "C:/Users/skia/Downloads/sample/test2"]
    # path_list = glob.glob(f"{path_root}/**", recursive=True)  
    # 'C:/Users/skia/Downloads/sample\\', 'C:/Users/skia/Downloads/sample\\test1', 'C:/Users/skia/Downloads/sample\\test1\\test1.nrrd', 'C:/Users/skia/Downloads/sample\\test2', 'C:/Users/skia/Downloads/sample\\test2\\test1.nrrd'
    path_list = [ f.path for f in os.scandir(path_root) if f.is_dir() ]
    return path_list

path_ex = "C:/Users/skia/Downloads/sample/test2"

path_root = "Z:/Dataset/3dSlicer/HeadCT/alkee-testing"
paths = get_all_child_path(path_root)
for path in paths:
    if not contains_same_dir_name_nnrd(path):
        print(path)

####################################
# nrrd_path = get_nrrd_path(path_ex)
# search(path_ex)
# print(nrrd_path)
# 두 경로를 비교할 수는 없었다
#C:/Users/skia/Downloads/sample/test2\test1.nrrd
#C:/Users/skia/Downloads/sample/test2/test2.nrrd
# /를 \로 바꾸었다


