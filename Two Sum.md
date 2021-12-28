# Question: #1 Two Sum 
#### 2021/12/27  


## Problem-solving steps
* 문제 해석: 주어진 배열과 정수로부터 인덱스들을 리턴해라 
타겟이 정수임 배열에서 숫자를 뽑아서 타겟이 되면! 해당 인덱스를 리턴
* 알고리즘: 버블정렬?
* 자료구조: 모르겠음
* 문제 해결 과정: 
배열의 i를 for 돌리기
j도 넣어 이중 for문 만들어 모든 경우의 수를 뽑기 nCr안 쓸거임
모든 경우의 수 i + j 가 만약 타겟과 같다면
i와 j를 리턴


---

첫번째 답 (8분걸림)
```python3
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        for i,v1 in enumerate(nums):
            for j, v2 in enumerate(nums):
                result = nums[i] + nums[j]
                if result == target:
                    return [i,j]
```

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        for i,v1 in enumerate(nums):
            for j, v2 in enumerate(nums):
                result = nums[i] + nums[j]
                if nums[j] == target - nums[i]:
                    return [i,j]
```

시간부족으로 더 만들지 못 함. 
선생님 답변을 보며 수정할 부분
