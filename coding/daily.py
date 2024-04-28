
from collections import defaultdict
import bisect

class Solution:
    def baseNeg2(self, n: int) -> str:
        ans = ""
        while(n):
            remain = int(n%(-2))
            ans = ans + chr(ord('0')+abs(remain))
            if remain < 0:
                n = int(n/(-2))+1
            else:
                n = int(n/(-2))
        if ans == "":
            return "0"
        else:
            return ans.reverse()
        
        
if __name__ == '__main__':
    s1 = Solution()
    print(s1.baseNeg2(5))
        