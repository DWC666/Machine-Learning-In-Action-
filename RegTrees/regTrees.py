from numpy import *


def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine)) # 将每个元素映射为浮点型
        dataMat.append(fltLine)
    # 最后一列为目标变量
    return dataMat


def binSplitDataSet(dataSet, feature, value):
    '''
    @Param:
    dataSet -> 数据集合
    feature -> 待切分的特征
    value -> 待切分特征的某个值
    '''
    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1


def regLeaf(dataSet):
    '''
    求叶节点模型，在回归树中即为目标变量的均值
    '''
    return mean(dataSet[:, -1])


def regErr(dataSet):
    '''
    计算连续型数据的混乱度：平方误差的总值，可以通过求均方误差乘以样本点个数得到    
    '''
    return var(dataSet[:, -1]) * shape(dataSet)[0]


def chooseBestSplit(dataSet, leafFunc=regLeaf, errFunc=regErr, ops=(1, 4)):
    '''
    寻找最佳的二元切分方式
    '''
    # 容许的误差下降值
    tolS = ops[0]
    # 切分的最小样本数
    tolN = ops[1]
    # 如果目标变量的值都相等，则直接生成叶节点并退出
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafFunc(dataSet)
    m, n = shape(dataSet)
    # 当前数据集的混乱度（平方误差的总和）
    S = errFunc(dataSet)
    # 当前最小混乱度
    minS = inf
    # 最佳切分特征的索引
    bestIndex = 0
    # 最佳切分特征的值
    bestVal = 0
    # 遍历所有特征列
    for featIndex in range(n-1):
        # 遍历所有特征值
        for splitval in set(dataSet[:, featIndex].T.A.tolist()[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitval)
            # 如果切分得到的数据集过小，则跳过此轮循环
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
                continue
            newS = errFunc(mat0) + errFunc(mat1)
            if newS < minS:
                bestIndex = featIndex
                bestVal = splitval
                minS = newS
    # 如果混乱度减小不明显，则直接创建叶节点并退出
    if (S - minS) < tolS:
        return None, leafFunc(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestVal)
    # 如果切分出的数据集很小，则直接创建叶节点并退出
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        return None, leafFunc(dataSet)
    return bestIndex, bestVal


def createTree(dataSet, leafFunc=regLeaf, errFunc=regErr, ops=(1, 4)):
    '''
    递归创建回归树

    @Param:
    dataSet -> 数据集合
    leafFunc -> 建立叶节点的函数
    errFunc -> 误差计算函数
    ops -> 构建数所需的其他参数
    '''
    feat, val = chooseBestSplit(dataSet, leafFunc, errFunc, ops)
    # 满足停止条件时返回叶节点值
    if feat == None:
        return val
    retTree = {}
    retTree["splitIndex"] = feat
    retTree["splitVal"] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree["left"] = createTree(lSet, leafFunc, errFunc, ops)
    retTree["right"] = createTree(rSet, leafFunc, errFunc, ops)
    return retTree


def isTree(obj):
    '''
    判断输入变量是否为一棵树（或，是否叶节点），返回布尔类型
    '''
    return (type(obj).__name__ == "dict")


def getMean(tree):
    '''
    递归函数，从上往下遍历树直到遇见叶节点。如果找到两个叶节点，则计算它们的平均值。
    该函数对树进行塌陷处理（即返回树的平均值）
    '''
    if isTree(tree["right"]):
        tree["right"] = getMean(tree["right"])
    if isTree(tree["left"]):
        tree["left"] = getMean(tree["left"])
    return (tree["left"] + tree["right"]) / 2.0


def prune(tree, testDate):
    '''
    通过比较当前两个叶节点合并前后的误差，对回归树进行后剪枝
    '''
    # 没有测试数据则对树进行塌陷处理
    if shape(testDate)[0] == 0:
        return getMean(tree)
    if (isTree(tree["right"]) or isTree(tree["left"])):
        lSet, rSet = binSplitDataSet(testDate, tree["splitIndex"], tree["splitVal"])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
        # 如果左右分支都是叶节点，则判断是否可以合并
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testDate, tree["splitIndex"], tree["splitVal"])
        # print("lSet[:, -1]: ", lSet[:, -1])
        # print("tree['left']: ", tree['left'])
        # print(lSet[:, -1] - tree['left'])
        # print('\n')
        # 合并前的误差
        errorNoMerge = sum(power(lSet[:, -1] - tree['left'], 2)) +\
                       sum(power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right']) / 2.0
        # 合并后的误差
        errorMerge = sum(power(testDate[:, -1] - treeMean, 2))
        if errorMerge < errorNoMerge:
            print("merging")
            return treeMean
        else:
            return tree
    else:
        return tree


def linearSolve(dataSet):
    '''
    获取输入数据集对应的简单线性回归模型
    '''
    m, n = shape(dataSet)
    X = mat(ones((m, n)))
    Y = mat(ones((m, 1)))
    # X的第0列默认为全为1
    X[:, 1:n] = dataSet[:, 0:n-1]
    Y = dataSet[:, -1]
    xTx = X.T * X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws, X, Y


def modelLeaf(dataSet):
    '''
    模型树：与一般回归树在叶节点中存储常数值不同，模型树在叶节点中存储一个拟合当前数据集的线性模型,
            模型树的优点在于更好的解释性和更高的预测精度。
    @return:
        返回模型树中叶节点中线性模型的回归系数
    '''
    ws, X, Y = linearSolve(dataSet)
    return ws


def modelErr(dataSet):
    '''
    返回预测值与真实值的平方误差之和
    '''
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat, 2))


def regTreeEval(model, inDat):
    '''
    对回归树的叶节点进行预测，直接返回叶节点中存储的常数值
    '''
    return float(model)


def modelTreeEval(model, inDat):
    '''
    对模型树的叶节点进行预测，计算并返回预测值
    '''
    n = shape(inDat)[1]
    X = mat(ones((1, n+1)))
    X[:, 1:n+1] = inDat
    return float(X * model)


def treeForeCast(tree, inDat, modelEval=regTreeEval):
    '''
    自顶向下遍历整棵树，直到遇见叶节点。
    对回归树的叶节点进行预测，则调用regTreeEval()，modelEval的默认值是regTreeEval;
    对模型树的叶节点进行预测，则调用modelTreeEval()。

    @Param:
        tree -> 树模型
        inDat -> X向量（特征向量）
        modelEval -> 叶节点预测函数
    '''
    # 递归出口
    # 如果不是树，则返回该节点的预测值
    if not isTree(tree):
        return modelEval(tree, inDat)
    # 如果数据中切分特征的值大于树中切分特征值，则应该划分到左分支，否则应该划分到右分支
    if inDat[tree["splitIndex"]] > tree["splitVal"]:
        # 如果左分支仍然是树，继续递归
        if isTree(tree["left"]):
            return treeForeCast(tree["left"], inDat, modelEval)
        # 如果右分支是叶节点，则返回叶节点预测值
        else:
            return modelEval(tree["left"], inDat)
    else:
        if isTree(tree["right"]):
            return treeForeCast(tree["right"], inDat, modelEval)
        else:
            return modelEval(tree["right"], inDat)


def createForeCast(tree, testData, modelEval=regTreeEval):
    '''
    以向量形式返回一组预测值
    @Param:
        tree -> 树模型
        testData -> 特征矩阵
        modelEval -> 叶节点预测函数
    '''
    m = len(testData)
    yHat = mat(zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat





if __name__ == "__main__":
    trainMat = mat(loadDataSet("bikeSpeedVsIq_train.txt"))
    testMat = mat(loadDataSet("bikeSpeedVsIq_test.txt"))
    tree = createTree(trainMat, modelLeaf, modelErr, ops=(1, 20))
    yHat = createForeCast(tree, testMat[:, 0], modelEval=modelTreeEval)
    coef = corrcoef(yHat, testMat[:, 1], rowvar=0)

    print(coef)
    # testData = loadDataSet("ex2test.txt")
    # testData = mat(testData)
    # regTree = prune(tree, testData)
    # print(regTree)