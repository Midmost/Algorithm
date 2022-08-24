# 폴더를 돌면서 자동으로 체크해줘


###############################################################################################

# exec(open("Z:/Dataset/Label/temporal,facial artery/validation.py").read())

# 1.볼륨명과 세그멘테이션명은 같아야 한다. 
# 2.세그멘테이션 안의 레이블명은 지정한 이름으로 저장되어야 한다. 

import slicer
import glob

def get_volume_name():
    vol_node = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLScalarVolumeNode')
    vol_name = vol_node.GetName()
    
    return vol_name

def get_segment_name():
    seg_node = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLSegmentationNode') 
    seg_name = seg_node.GetName()
    return seg_name

def has_same_name():
    """볼륨명과 세그멘테이션명이 같은 지 확인하는 함수

    Returns: 같을 경우 True, 다르면 False 
    """
    # 볼륨명과 세그멘테이션명은 같아야 한다. 
    if get_volume_name() == get_segment_name():
        return True
    # else를 쓸 이유가 없는게 위에서 리턴하는 게 true 면 어차피 false인 경우만 아래로 내려오니까.
    # 이게 더 좋다 nested 하지 않게 if문을 만들기
    return False

######################################################################################

# 라벨을 찾고 지정된 이름이 아닌 경우 출력하기 
# 지정된 이름 리스트 ["temporal_artery", "temporal_vein", "facial_artery", "facial_vein"]

# TODO: ["temporal_artery","temporal_artery"] 인 경우 추가하기
labels = ["temporal_artery", "facial_artery", "temporal_vein", "facial_vein", 
          "temporal_artery_right", "temporal_artery_left","facial_artery_right", "facial_artery_left",
          "temporal_vein_right", "temporal_vein_left","facial_vein_right", "facial_vein_left",
          "lingual_artery","NA"]
#라벨을 찾는다
def find_label():
    find_list = []
    segNode = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLSegmentationNode')
    segmentation = segNode.GetSegmentation()
    segIds = segmentation.GetSegmentIDs()
    for segId in segIds:
        segment = segmentation.GetSegment(segId)
        find_list.append(segment.GetName())
    return find_list  # [temporal_artery, df]

def get_invalid_labels():
    invalid_labels = []
    found_list = find_label()
    for label in found_list:
        if label not in labels:
            invalid_labels.append(label)
    return invalid_labels

# 씬이 로드되어있다고 가정한 상태에서 작업을 했었기 때문에 
# 씬이 로드될 수 있게끔 하는 함수가 필요한 거였음.
def load(slicer_scene_path: str):
    slicer.mrmlScene.Clear()
    slicer.util.loadScene(slicer_scene_path)
    return

# exit("Z:/Dataset/3dSlicer/HeadCT/alkee-testing/29539554-75785777-7-CAC/2022-08-01-Scene.mrml")

# 
def get_all_scenes(root_dir: str):
    # root_dir = "Z:/Dataset/3dSlicer/HeadCT/Reviewing"
    output = glob.glob(root_dir + "/**/*.mrml", recursive=True)
    return output

# print(get_all_scenes("Z:/Dataset/3dSlicer/HeadCT/alkee-testing"))  # ['Z:/Dataset/3dSlicer/HeadCT/alkee-testing\\29539554-75785777-7-CAC\\2022-08-01-Scene.mrml', 'Z:/Datase

# for scene_path in get_all_scenes("Z:/Dataset/3dSlicer/HeadCT/alkee-testing"):
#     #load함수를 바로 호출하지 말고
#     # 0번째 인덱스로 씬이 로드가 되면
#     # invalid_labels를 확인한다음 프린트하고 
#     # exitScene을 한다 
#     # 나머지 인덱스도 같은 방식으로 반복

#     load(scene_path)
#     invalid_labels = get_invalid_labels()
#     if len(invalid_labels) != 0:
#         print(f"찾은 라벨들이 지정된 라벨들에 없음 {invalid_labels}")
#     slicer.util.saveScene(scene_path)
#     slicer.util.quit()

correct= 'Z:/Dataset/3dSlicer/HeadCT/alkee-testing/29539554-75785777-7-CAC/2022-08-01-Scene.mrml'
incorrect = 'Z:/Dataset/3dSlicer/HeadCT/alkee-testing/50829509-HE0222-5-CAO/2022-01-19-Scene.mrml'
scene_file_paths = get_all_scenes("Z:/Dataset/3dSlicer/HeadCT/Reviewing")

for scene_file_path in scene_file_paths:
    
    try:
        slicer.app.processEvents()
        load(scene_file_path)
        
        # 지정된 라벨을 갖고 있지 않은 씬을 모두 출력하기
        invalid_labels = get_invalid_labels()
        if len(invalid_labels) != 0:
            # 씬은 이미 불러와져 있고
            print(invalid_labels)
            print(scene_file_path)
        
    except Exception as e:
        print(f'{scene_file_path} : {e}')
        
        
    # 씬이 불러와졌어 그다음에 하고 싶은건?
    # 그 씬의 라벨을 확인하고 싶어.
    # 그래서 그 씬의 라벨이 지정된 라벨과 다르다면
    # 그 씬의 경로를 출력하고 싶어
    # 씬의 라벨이 지정된 라벨안에 포함된다면 출력을 안 하게 하고 싶어

    






###############################
# 상황들을 생각할 수 있어야 함. 엇 이게 이거를 커버하지 못 하네>>>?
# 해놓고 고치고 이걸 반복하렴녀 코드가 간결하고 읽기 쉬워야 그게 쉬워짐

# 어떻게는 나중에 뭐가 필요한 지 생각하기
# 어떤 규칙이 있는 데 그게 지켜졌는 지를 확인하고 싶다
# 어떤 규칙들이 있는 지 먼저 생각하기 
# 세그멘테이션 데이터 명 규칙 
