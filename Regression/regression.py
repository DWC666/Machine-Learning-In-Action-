import numpy as np
import matplotlib.pyplot as plt


def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) - 1 #get number of fields 
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


def standRegres(xArr, yArr):
    '''
    标准线性回归，使用最小均方误差求回归系数
    '''
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    # 若矩阵的行列式为0，则不能求逆
    if np.linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    # 回归系数
    ws = xTx.I * (xMat.T * yMat)
    return ws


def lwlr(testPoint, xArr, yArr, k=1.0):
    '''
    局部加权线性回归(Locally Weighted Linear Regression)：给待预测点附近的每个
    点赋予一定权重，离待预测点越近，权重越大
    '''
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye(m))
    for i in range(m):
        diffMat = testPoint - xMat[i, :]
        weights[i, i] = np.exp(diffMat * diffMat.T / (-2 * (k**2)))
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    #求回归系数
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws


def lwlrTest(testArr, xArr, yArr, k=0.01):
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat


def plot(xMat, y, yHat):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 将xMat的第二列展开成一维。再转化为np.array类型
    ax.scatter(xMat[:, 1].flatten().A[0], y)
    # 不对数据进行排序的话，绘图将出错
    # 将xMat的第二列排序并取索引
    srtIndex = xMat[:, 1].argsort(0)
    # 对xMat排序；[:, 0, :]将三维矩阵转化为二维
    xSort = xMat[srtIndex][:, 0, :]
    # xCopy = xMat.copy()
    # yHat = xMat * ws
    print(yHat.shape, yHat[srtIndex].shape)
    ax.plot(xSort[:, 1], yHat[srtIndex], 'r')
    plt.show()


def rssError(yArr, yHatArr):
    return ((yArr - yHatArr)**2).sum()


def ridgeRegres(xMat, yMat, lam=0.2):
    '''
    当特征比样本点还多(n > m)时，意味着特征矩阵X不是满秩矩阵，无法求逆，
    此时可以通过岭回归，即在xTx上加上lambda * I
    (lambda为自定义参数，惩罚项；I为单位矩阵)，使得矩阵非奇异。
    lambda非常小时，等同于普通回归。
    '''
    xTx = xMat.T * xMat
    # print(xMat.shape, xTx.shape)
    denom = xTx + np.eye(xMat.shape[1]) * lam
    if np.linalg.det(denom) == 0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T * yMat)
    return ws


def ridgeTest(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    print(yMat.shape)
    # 数据标准化，均值为0，方差为1
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean #to eliminate X0 take mean off of Y
    xMeans = np.mean(xMat, 0)
    xVar = np.var(xMat, 0)
    xMat = (xMat - xMeans) / xVar

    numTest = 30
    wMat = np.zeros((numTest, xMat.shape[1]))
    for i in range(numTest):
        ws = ridgeRegres(xMat, yMat, np.exp(i-10))
        wMat[i, :] = ws.T
    return wMat


def regularize(xMat):
    '''
    regularize by columns
    按列标准化
    '''
    inMat = xMat.copy()
    inMeans = np.mean(inMat, 0)   #calc mean then subtract it off
    inVar = np.var(inMat, 0)      #calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar
    return inMat


def stageWise(xArr, yArr, eps=0.01, numIt=100):
    '''
    前向逐步回归算法：效果与lasso差不多，但计算更简单，
    属于贪心算法，即每一步都尽可能减少误差

    @Param:
    eps:系数调整步长
    numIt:迭代次数
    '''
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    # 标准化，均值为0，方差为1
    xMat = regularize(xMat)
    m, n = xMat.shape
    returnMat = np.zeros((numIt, n))
    ws = np.zeros((n, 1))
    wsTest = ws.copy()
    wsBest = ws.copy()
    for i in range(numIt):
        print(ws.T)
        # 设置当前最小误差为正无穷
        lowestError = np.inf
        for j in range(n):
            # 对每个特征增大或减少，看是否可以减少误差
            for sign in [-1, 1]:
                wsTest = ws.copy()
                # 改变一个系数得到一个新的W向量
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                # 计算新W下的误差
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsBest = wsTest
        ws = wsBest.copy()
        returnMat[i, :] = ws.T
    return returnMat





if __name__ == "__main__":
    xArr, yArr = loadDataSet("abalone.txt")
    # ws = standRegres(xArr, yArr)
    # yHat01 = lwlrTest(xArr[0:99], xArr[0:99], yArr[0:99], 0.1)
    # yHat1 = lwlrTest(xArr[0:99], xArr[0:99], yArr[0:99], 1)
    # yHat10 = lwlrTest(xArr[0:99], xArr[0:99], yArr[0:99], 10)
    # print(rssError(yArr[0:99], yHat01.T))
    # print(rssError(yArr[0:99], yHat1.T))
    # print(rssError(yArr[0:99], yHat10.T))
    # plot(np.mat(xArr[0:99]), yArr[0:99], yHat1)

    # ridgeWeights = ridgeTest(xArr, yArr)
    # print(ridgeWeights.shape, '\n', ridgeWeights)
    
    weightsMat = stageWise(xArr, yArr, 0.005, 1000)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(weightsMat)
    plt.show()
