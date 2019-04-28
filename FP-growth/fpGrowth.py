"""
FP-growth算法是一种用于发现数据集中的频繁模式的方法，只对数据集扫描两次，执行速度快。
在FP-growth算法中，数据集存储在一个被称为FP(freqent pattern)树的结构中。
FP树构建完之后，可以通过查找元素项的 条件基 及构建 条件FP树 来发现频繁模式。
该过程不断以更多元素作为条件重复进行，直到条件FP树中没有元素为止。
"""

class treeNode:
    '''
    FP树节点类：
        name: 节点名
        count: 计数值
        nodeLink: 链接相似的元素项
        parent: 当前节点的父节点
        children: 当前节点的子节点
    '''
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}

    # 增加计数值
    def inc(self, numOccur):
        self.count += numOccur

    # 递归显示节点及子节点
    def disp(self, ind=1):
        print('  '*ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind+1)


def createTree(dataSet, minSupport=1):
    '''
    @param:
        dataSet: 字典类型，键为项集(frozenset类型)，值为项集对应的频率
        minSupport: 最小支持度
    '''
    headerTable = {}  # 头指针表，存储各元素的频率
    # go over dataSet twice
    for trans in dataSet:  # first pass counts frequency of occurance
        for item in trans:
            # 统计每个元素项出现的频率
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    for k in list(headerTable.keys()):  # remove items not meeting minSupport
        if headerTable[k] < minSupport:
            del(headerTable[k])

    freqItemSet = set(headerTable.keys())
    # print("freqItemSet: ", freqItemSet)
    if len(freqItemSet) == 0:  # if no items meet min support -->get out
        return None, None

    for k in headerTable:
        # 对头指针表进行扩展，以保存计数值和该类的第一个元素项的指针
        headerTable[k] = [headerTable[k], None]  # reformat headerTable to use node link

    retTree = treeNode('Null Set', 1, None)  # create root node of the tree
    for tranSet, count in dataSet.items():  # go through dataset 2nd time
        localD = {}  # 局部变量，保存频繁项及其计数值
        # 注意：tranSet是frozenSet类型，是无序的。
        # 后面对于相同key的元素每次排序结果不一样，故得到的树每次结果不一样
        for item in tranSet:  # put transaction items in order
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
                # print('item: ', item, 'count: ', headerTable[item][0])
        if len(localD) > 0:
            # 基于计数值对频繁项进行降序排序
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
            # print(orderedItems)
            # print(orderedItems)
            updateTree(orderedItems, retTree, headerTable, count)  # populate tree with ordered freq itemset
    return retTree, headerTable


def updateTree(items, inTree, headerTable, count):
    '''
    从根节点开始构建树。
    对于每一个项集中的第一个元素，如果其已经在根节点的子节点中，则增加该子节点的计数值；
    否则，将该元素添加为根节点的新子节点。然后，继续递归遍历项集中的其余元素。

    @param:
        items: 每一事务实例的频繁项列表，list类型
        inTree: 构建树时的起始节点，即父节点
        headerTabel: 头指针表，字典类型，存储频繁元素对应的频率及第一个实例的指针
        count: 每一事务实例的计数值
    '''
    if items[0] in inTree.children:  # check if orderedItems[0] in root.children
        inTree.children[items[0]].inc(count)  # increament count
    else:  # 如果items[0] 不在根节点的子节点中，则添加至子节点
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        if headerTable[items[0]][1] is None:  # 如果头指针表中该类的第一个元素为空，则更新头指针表
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:  # 如果不为空，则添加至链表末尾
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:  # call updateTree() with remaining ordered items
        updateTree(items[1::1], inTree.children[items[0]], headerTable, count)


def updateHeader(nodeToTest, targetNode):
    # 从头指针表的nodeLink开始，一值遍历到链表末尾
    while nodeToTest.nodeLink is not None:
        nodeToTest = nodeToTest.nodeLink
    # 将目标节点添加到链表末尾
    nodeToTest.nodeLink = targetNode


def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat


def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict


def ascendTree(leafNode, prefixPath):
    # 从某一节点上溯到根节点，并保存途经的节点
    if leafNode.parent is not None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)


def findPrefixPath(basePat, treeNode):
    '''
    参数basePat在函数中并没有使用到
    '''
    # 条件模式基(conditional pattern base)：
    # 从根节点到目标节点的路径集合（不包括根节点和目标节点）。
    # 每条路径对应一个计数值，为目标节点的计数值
    condPats = {}
    while treeNode is not None:  # 遍历链表直到末尾
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats


def mineTree(headerTable, minSupport, preFix, freqItemList):
    '''
    @param:
        headerTabel: 头指针表，字典类型，存储频繁元素对应的频率及第一个实例的指针
        minSupport: 最小支持度，int类型
        preFix: 前缀，set()类型
        freqitemList: 频繁项列表，list()类型
    '''
    # 基于计数值对头指针表从小到大排序， bigL保存所有频繁项
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1][0])]
    for basePat in bigL:  # 从频率最小的元素开始遍历
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        # 利用频繁项的条件模式基构建条件FP树
        myCondTree, myHead = createTree(condPattBases, minSupport)
        
        if myHead is not None:  # 如果条件FP树不为空
            print('conditional tree for: ', newFreqSet)
            myCondTree.disp()
            # 在当前频繁项集newFreqSet的基础上构建条件FP树，产生更复杂的频繁项集
            mineTree(myHead, minSupport, newFreqSet, freqItemList)


if __name__ == "__main__":
    data = loadSimpDat()
    initSet = createInitSet(data)
    tree, headerTable = createTree(initSet, 3)
    # tree.disp()
    condPats = findPrefixPath('r', headerTable['r'][1])
    # print(condPats)
    freqItems = []
    mineTree(headerTable, 3, set([]), freqItems)
    print(freqItems)

    # parseDat = [line.split() for line in open('kosarak.dat').readlines()]
    # initSet = createInitSet(parseDat)
    # tree, headerTable = createTree(initSet, 100000)
    # freqList = []
    # mineTree(headerTable, 100000, set([]), freqList)
    # print(freqList)