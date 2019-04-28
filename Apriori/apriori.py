from numpy import *


def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


def createC1(dataSet):
    # C1是大小为1的所有候选项集的集合
    C1 = []
    # 遍历数据中的每一个交易记录
    for transaction in dataSet:
        # 遍历交易中的每一项
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    # 排序
    C1.sort()
    # 对C1中的每个列表项构建一个不变集合，这样可以作为字典中的key
    return list(map(frozenset, C1))


def scanD(D, Ck, minSupport):
    '''
    @param:
        D -> 数据集 [set, set, ...]
        Ck -> 候选项集 [frozenset, frozenset, ...]
        minSupport -> 最小支持度 float
    '''
    # 存储Ck中每项对应的计数值
    ssCnt = {}
    for tid in D:
        for can in Ck:
            # 如果候选项是交易记录的子集
            if can.issubset(tid):
                if can not in ssCnt:
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    # 交易记录总数
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        # 计算支持度
        support = ssCnt[key] / numItems
        # 如果大于最小支持度，则插入返回列表的头部
        if support >= minSupport:
            retList.append(key)
        # 存储频繁项集的支持度
        supportData[key] = support
    return retList, supportData


def aprioriGen(Lk, K):
    '''
    @param:
        Lk -> 候选项集合列表，k代表每个候选项集合中元素个数
        K -> 将要生成的候选项集合中元素个数，K=k+1
    函数说明：函数利用某一候选项集合的列表生成另一个候选项集合列表，使得集合中元素个数加1
    '''
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            # 取集合中前K-2个元素进行比较
            L1 = list(Lk[i])[: K-2]
            L2 = list(Lk[j])[: K-2]
            L1.sort()
            L2.sort()
            # 如果前K-2个元素相等，则将两集合合并
            # 这样可以生成一个含有K个不同元素的集合
            # 比较前K个元素而不是直接合并集合的目的在于避免生成相同集合，减少计算量
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList


def apriori(dataSet, minSupport=0.5):
    '''
    函数说明:生成所有频繁项集合
    Apriori原理：如果一个元素项是不频繁的，那么包含该元素的超集也是不频繁的。据此可以减少计算量。
    '''
    # 生成只含有单个元素的候选集合
    C1 = createC1(dataSet)
    D = list(map(set, dataSet))
    # 过滤低于最小支持度的集合
    L1, supportData = scanD(D, C1, minSupport)
    # L存储所有频繁项集合，即L1, L2, L3...
    L = [L1]
    k = 2
    # 循环生成L2, L3, L4...
    # 直到Ln为空，说明已找到所有频繁项集合
    while len(L[k-2]) > 0:
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)
        # 更新支持度字典
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData


def generateRules(L, supportData, minConf=0.7):  #supportData is a dict coming from scanD
    '''
    函数说明：生成所有满足可信度要求的关联规则
    @param:
        L: 存储所有频繁项集合，即L1, L2, L3...
        supportData: 字典类型，存储频繁项的可信度
        minConf: 最小可信度
    '''
    bigRuleList = []
    for i in range(1, len(L)):  #only get the sets with two or more items
        for freqSet in L[i]:
            # 创建只包含单个元素集合的列表，如：{0, 1, 2} --> [{0}, {1}, {2}]
            H1 = [frozenset([item]) for item in freqSet]
            if i > 1:  #如果频繁项的元素数目大于2，则进一步合并
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:  # 如果频繁项只含有2个元素，则计算可信度
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList


def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    '''
    计算规则的可信度
    @param:
        freqSet: 规则中前件和后件的并集
        H: 后件集合
        brl: 即bigRuleList, 存储所有满足可信度要求的规则的信息列表
        minConf: 最小可信度
    '''
    prunedH = []
    for conseq in H:
        # 规则 P->H (P称为前件，H称为后件)的可信度为：support(P | H) / support(P)
        conf = supportData[freqSet] / supportData[freqSet-conseq]  #calc confidence
        if conf >= minConf:
            print(freqSet-conseq, '-->', conseq, 'conf: ', conf)
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)  # 将满足可信度要求的规则中的后件存入列表
    return prunedH


def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    '''
    函数说明：从初始项集中生成更多的关联规则
    @param:
        freqSet: 规则中前件和后件的并集
        H: 后件集合
        brl: 即bigRuleList, 存储所有满足可信度要求的规则的信息列表
        minConf: 最小可信度
    '''
    m = len(H[0])  # 计算H中频繁项的大小
    if len(freqSet) > (m + 1): # 如果频繁项大到可以移除大小为m+1的子集
        Hmp1 = aprioriGen(H, m+1)  # 生成H中元素的无重复组合, 每个组合大小为m+1
        # 筛选满足可信度要求的组合，这些组合包含所有可能的规则，即代表 P-->H 中的H（后件），大小为m+1
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if len(Hmp1) > 1:  # 如果不止一条规则满足要求，则尝试进一步组合这些规则得到新规则
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)

if __name__ == "__main__":
    data = loadDataSet()
    L, support = apriori(data, 0.5)
    # print(L)
    rules = generateRules(L, support, 0.5)
    print(rules)