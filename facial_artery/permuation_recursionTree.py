class Solution:
        
    S = 'ABC'
    
    def permute(self, curr, S, N,words =[]):
        
        if(len(curr) == N):
            words.append(curr)
            return
        for ch in S:
            self.permute(curr+ch, S.replace(ch,""), N, words)
            
    def find_permutation(self, S):
        words = []
        N = len(S)
        self.permute("", ''.join(sorted(S)), N, words)
        return words