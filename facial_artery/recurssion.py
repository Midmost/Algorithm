def recursive_function(i):
    if i == 3:
        return
    print(i, '번째 재귀함수에서', i+1, '번째 재귀함수를 호출합니다')
    recursive_function(i + 1)
    print(i, '번째 재귀함수를 종료')
    
    
recursive_function(1)

#생각했던 프린트는
# 1번 째 재귀함수에서 2번째 재귀함수를 호춣합니다
# 1번 째 재귀함수를 종류
# 2번 째 재귀함수에서 3번째 재귀함수를 호출합니다
# 2번 째 재귀 함수를 종료... 이렇게 나아갈 줄 알았는데...

# 디버깅을 돌려보니 어떻게 작동하는지는 알겠어
# 그런데 if문 해당되어서 들어가서 리턴까지 했는데
# 어떻게 다시 5번 줄로 가서 진행이 되는 거지?? 
# 함수밖으로 탈출해야 하는 거 아닌가?