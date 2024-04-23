

def quicksort(testlist):
    if len(testlist) == 1:
        return testlist
    tmp_l = []
    tmp_r = []
    cnt = testlist[0]
    for i in range(len(testlist)):
        if testlist[i] > cnt :
            tmp_l.append(testlist[i])
        else:
            tmp_r.append(testlist[i])
    if len(tmp_l) == 0:
        return  tmp_r
    if len(tmp_r)== 0:
        return tmp_l
    if tmp_l[-1]<tmp_r[0]:
        return quicksort(tmp_r) + quicksort(tmp_l)
    return quicksort(tmp_l) + quicksort(tmp_r)

def spiralOrder(matrix):
    book = set()
    dirt = [[0,1],[1,0],[0,-1],[-1,0]]
    dirt_tmp = 0
    result = []
    w , h = len(matrix),len(matrix[0])
    tmp = 1
    nw , nh = 0,0
    result.append(matrix[nw][nh])
    book.add((0,0))
    while tmp < h * w:
        if dirt_tmp == 4:
            dirt_tmp = 0
        
        for i in range(max(h,w)):
            tw = nw + dirt[dirt_tmp][0]
            th = nh + dirt[dirt_tmp][1]
            if tw >= 0 and tw < w and th >= 0 and th < h and (tw,th) not in book:
                book.add((tw,th))
                nw = tw
                nh = th
                result.append(matrix[tw][th])
                tmp = tmp + 1
            else:
                break

        dirt_tmp = dirt_tmp + 1
    
    return result


def restoreIpAddresses(s):
    
    result = []

    def judge(tmp):
        for i in tmp:
            if int(i)>255:
                return False
            if int(i[0]) == '0' and len(i) > 1:
                return False
        return True
    
    def getresult(s,book,tmp):
        if book == 4 and len(tmp) == 4:
            if len(s) == 0 and judge(tmp):
                ans = ''
                for i in tmp:
                    ans = ans + str(i) + '.'
                result.append(ans[:-1])
            return
            
        for i in range(1,4):
            if len(s) >= i:
                tmp.append(s[:i])
                getresult(s[i:],book+1,tmp)
                tmp = tmp[:-1]
    
    getresult(s,0,[])

    return result

class Solution:
    def str2int(self,s:str) -> int:
        result = 0
        lenth = len(s)
        cnt = 1
        for i in range(lenth):
            result = result * 10 + ord(s[i]) - ord('0')
        return result
            

    def myAtoi(self, s: str) -> int:

        maxInt = pow(2,31)
        result = 0
        start,end = -1,-1
        lenth = len(s)
        postive = 0
        if lenth == 0:
            return 0

        for i in range(lenth):
            if s[i] == '-':
                postive = 1
            elif ord(s[i]) >= ord('0') and ord(s[i]) <= ord('9'):
                start = i
                break
            elif s[i] == ' ' or s[i] == '+':
                continue
            elif start == -1:
                return 0
        for i in range(start,lenth):
            cnt = ord(s[i])
            if cnt >= ord('0') and cnt <= ord('9'):
                end = i
                continue
            else:
                end = i-1
                break
        result = self.str2int(s[start:end+1])
        if result > maxInt:
            result = maxInt
        if postive == 1:
            return -result
        else:
            return result
        




if __name__ == '__main__':
    
    s1 = Solution()
    print(s1.myAtoi("+-12"))
    
    # print(quicksort(test))
