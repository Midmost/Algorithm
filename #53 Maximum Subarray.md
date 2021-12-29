# Question: #53 Maximum Subarray
#### 2021/12/28


## Problem-solving steps
* 문제 해석: array의 가장 큰 수를 기준으로 인접한 수를 양쪽으로 더해가다 합친 수가 적어지면 그 전 수를 리턴하자.
* 알고리즘: array, dynamic programming, divide and conquer(?), 완전탐색
* 자료구조: array
* 문제 해결 과정: 
1. array의 모든 요소들을 다 더한 것을 sum이라고 하자.
2. 가장 끝 수 부터 하나씩 요소를 뺀 것을 temp 에 담는다
3. 가장 첫 수 부터 하나씩 요소를 뺀 것을 temp2에 담는다
4. if temp > sum 라면 temp를 리턴
5. if temp2 > sum 라면 temp2를 리턴
6. temp 와 temp2를 비교해서 더 큰 수를 리턴


~~ 2. subarray는 가장 큰 수를 포함하여 가장 작은수까지(미포함) 인접한 모든 요소들을 더한다. ~~




---
```python3
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        
        # array의 모든 요소들을 다 더한 것을 sum이라고 하자
        sums = 0;
        for i in nums:
            sums = i + sums
        
        # 가장 첫 수 부터 하나씩 요소를 뺀 것을 temp 에 담는다
        temp = nums;
        a = len(nums) - 1
        for j in nums:            
            left = temp.pop(0)
            sumLeft = sums - left
            # if left > sum 라면 temp를 리턴
            if sumLeft > sums:
                print(sumLeft)
                
            
        
        

# if temp2 > sum 라면 temp2를 리턴
# temp 와 temp2를 비교해서 더 큰 수를 리턴
₩₩₩            
            
            
        
            
        
