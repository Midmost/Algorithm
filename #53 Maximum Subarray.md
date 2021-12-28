# Question: #53 Maximum Subarray
#### 2021/12/28


## Problem-solving steps
* 문제 해석: array의 가장 큰 수를 기준으로 인접한 수를 양쪽으로 더해가다 합친 수가 적어지면 그 전 수를 리턴하자.
* 알고리즘: array, dynamic programming, divide and conquer(?), 완전탐색
* 자료구조: array
* 문제 해결 과정: 
1. array의 모든 요소들을 비교하여 가장 큰 수 a와 가장 작은 수 b를 찾는다.  
~ 2. subarray는 가장 큰 수를 포함하여 가장 작은수까지(미포함) 인접한 모든 요소들을 더한다. ~




---
```python3
