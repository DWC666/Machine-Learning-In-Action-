import numpy as np
import random

def loadDataSet():
    dataMat = []
    labelMat = []
    with open("F:\\codes\\python\\Machine-Learning-In-Action-python3\\LogisticRegress\\testSet.txt") as f:
        for line in f.readlines():
            line = line.strip().split()
            dataMat.append([1.0, float(line[0]), float(line[1])])
            labelMat.append(int(line[2]))

    return dataMat, labelMat

def sigmoid(x):
    # return 1.0/(1 + np.exp(-x))
    '''
    当x是一个非常小的负数时，exp(-x)会过大，导致溢出，下面进行优化：
    原式分子分母同乘exp(x)这个很小的数，可以防止数据溢出
    '''
    if x >= 0:
        return 1.0/(1 + np.exp(-x))
    else:
        return np.exp(x)/(1 + np.exp(x))


def gradAscent(dataMatIn, classLabels):
    dataMat = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose() #转置，此错误可忽略
    m, n = np.shape(dataMat)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMat * weights)  #sigmoid函数矢量化运算
        error = labelMat - h
        # "dataMat.transpose() * error"  ==  " error.transpose() * dataMat"
        weights = weights + alpha * dataMat.transpose() * error
    #将矩阵转换为数组返回
    return weights.getA()

"""
常规的梯度上升算法每次更新系数都需要计算整个数据集，数据量大时计算复杂度过高。
改进的随机梯度上升算法一次仅用一个样本来更新系数，属于在线学习算法。
"""
def stocGradAscent0(dataMatIn, labelMatIn, Iters=150):
    m, n = np.shape(dataMatIn)
    weights = np.ones(n)
    for j in range(Iters):
        dataIndex = list(range(m))
        for i in range(m):
            #步长动态调整，逐步缩小
            alpha = 4 / (1.0 + j + i) + 0.01
            #随机选取样本
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(np.sum(dataMatIn[randIndex] * weights))
            error = labelMatIn[randIndex] - h
            weights = weights + alpha * error * dataMatIn[randIndex]
            #删除已使用的样本索引
            del(dataIndex[randIndex])

    return weights


def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    #注意X0 = 1，已省略
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

def classifyVector(inX, weights):
    prob = sigmoid(np.sum(inX * weights))
    if prob > 0.5:
        return 1
    else:
        return 0

def colicTest():
    frTrain = open("F:\\codes\\python\\Machine-Learning-In-Action-python3\\LogisticRegress\\horseColicTraining.txt")
    frTest = open("F:\\codes\\python\\Machine-Learning-In-Action-python3\\LogisticRegress\\horseColicTest.txt")
    trainSet = []
    trainLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i])) #必须转换为浮点型,否则出错。下同
        trainSet.append(lineArr)
        trainLabels.append(float(currLine[-1]))
    trainWeights = stocGradAscent0(np.array(trainSet), trainLabels, 500)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        # print("lineArr:", type(lineArr), np.shape(lineArr), "   trainWeights:", type(trainWeights), np.shape(trainWeights))
        if int(classifyVector(np.array(lineArr), trainWeights)) != int(currLine[-1]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print("the error rate is: %f" % errorRate)
    return errorRate

def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()

    print("After %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))


if __name__ == "__main__":
    multiTest()
