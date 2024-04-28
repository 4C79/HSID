
from collections import defaultdict
import bisect

class SnapshotArray:

    def bi_find(self,vec,l,r,val):
        mid = (l + r)//2
        if vec[mid][0] == val:
            return mid
        elif vec[mid][0] > val:
            return self.bi_find(vec,l,mid-1,val)
        else:
            return self.bi_find(vec,mid+1,r,val)

    def __init__(self, length: int):
        self.snap_count = 0
        self.history = defaultdict(list)

    def set(self, index: int, val: int) -> None:
        self.history[index].append((self.snap_count,val))

    def snap(self) -> int:
        self.snap_count = self.snap_count + 1
        return self.snap_count - 1


    def get(self, index: int, snap_id: int) -> int:
  
        return self.history[index][snap_id][1]


# Your SnapshotArray object will be instantiated and called as such:
obj = SnapshotArray(3)
obj.set(0,5)
param_2 = obj.snap()
obj.set(0,6)
param_3 = obj.get(0,0)
print(param_3)