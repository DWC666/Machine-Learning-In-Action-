from numpy import *
import time 
'''
* : 对于矩阵类型(np.mat())来说执行矩阵乘法，对于np.array类型来说是对应位置相乘
np.multiply(A, B): 无论什么类型，都是A, B对应位置元素相乘，输出和A或B大小相同
'''
def loadData(fileName):
    dataMat = []
    labelMat = []
    with open(fileName) as fr:
        for line in fr.readlines():
            lineArr = line.strip().split()
            dataMat.append([float(lineArr[0]), float(lineArr[1])])
            labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

#在(0, m)的区间范围内随机选择一个除i以外的整数
def selectJrand(i, m):
    j = i
    while(j == i):
        j = int(random.uniform(0, m))
    return j

#保证a在区间[L, H]中
def clipAlpha(a, H, L):
    if a > H:
        a = H
    if a < L:
        a = L
    return a


#输入：数据矩阵dataMatIn，标签向量classLabels，常数C，容错率toler，最大迭代次数maxIter
#输出：超平面位移项b，拉格朗日乘子alpha
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMat = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    b = 0
    m, n = shape(dataMat)
    alphas = mat(zeros((m, 1))) # m*1 矩阵
    iter = 0
    while(iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            #模型f(x)计算值
            #推导见周志华《机器学习》公式6.12
            fXi = float(multiply(alphas, labelMat).T * (dataMat * dataMat[i, :].T)) + b
            Ei = fXi - float(labelMat[i]) #计算值与真实值的误差
            #误差很大，可以对该数据实例所对应的alpha值进行优化
            if (labelMat[i] * Ei < -toler) and (alphas[i] < C) or (labelMat[i] * Ei > toler) and (alphas[i] > 0):
                #在(0, m)的区间范围内随机选择一个除i以外的整数，即随机选择第二个alpha
                j = selectJrand(i, m)
                fXj = float(multiply(alphas, labelMat).T * (dataMat * dataMat[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                #不能直接 alphaIold = alphas[i]，否则alphas[i]和alphaIold指向的都是同一内存空间
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                #接下来需要看Plata的论文
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print("L==H")
                    continue
                # eta 是alphas[j]的最优修改量
                eta = 2.0 * dataMat[i,:] * dataMat[j, :].T - dataMat[i,:] *\
                      dataMat[i,:].T - dataMat[j,:] * dataMat[j,:].T
                #如果eta >= 0，退出for语句的当前迭代
                if eta >= 0:
                    print("eta >= 0")
                    continue
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                #保证alphas[j]在区间[L, H]里面
                alphas[j] = clipAlpha(alphas[j], H, L)
                #检查alpha[j]是否有较大改变，如果没有则退出当前迭代
                if(abs(alphas[j] - alphaJold) < 0.00001):
                    print("j not moving enough")
                    continue
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * \
                     dataMat[i,:] * dataMat[i,:].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * \
                     dataMat[i,:] * dataMat[j,:].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * \
                     dataMat[i, :] * dataMat[j, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * \
                     dataMat[j, :] * dataMat[j, :].T 

                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print("iter: %d, i: %d, pairs changed: %d" % (iter, i, alphaPairsChanged))
        #这个迭代思路比较巧妙，m次循环中每次误差均在精度要求之内时结束本次iter,进行下一次迭代
        #只有在所有数据集上遍历maxIter次，且不再发生任何alpha修改之后，才退出while循环
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        print("iteration number: %d" % iter)
    return b, alphas


def kernelTrans(X, A, kTup):
    '''
    X->特征矩阵
    A->某样本的特征行向量
    kTup->包含核函数信息的元组，元组第一个参数是核函数类型的字符串，第二个参数是函数可能需要的可选参数
    '''
    m, n = shape(X)
    K = mat(zeros((m, 1)))
    #线性核函数
    if kTup[0] == 'lin':
        K = X * A.T
    #径向基核函数
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = exp(K / (-1 * kTup[1] ** 2))
    else:
        raise NameError("There is a problem that Kernel is not recognized!!!")
    return K


class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        #如果C很大，分类器将力图通过分类超平面对所有样例正确分类；反之，分类间隔将尽可能大
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        #误差缓存：第一列是eCache是否有效的标志位(0或1)，第二列是实际的误差值E
        self.eCache = mat(zeros((self.m, 2)))
        self.K = mat(zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)

#计算预测值与真实值的误差
def calcEk(oS, k):
    fXk = float(multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b)
    Ek = fXk - float(oS.labelMat[k, :])
    return Ek


#选择具有最大步长的alphaJ
def selectJ(i, oS, Ei):
    maxK = -1 #最大步长对应的j
    maxDeltaE = 0 #最大步长
    Ej = 0 #j对应的误差
    oS.eCache[i] = [1, Ei]
    # .A表示将矩阵转化为array，nonzero()返回非零E值对应的alpha值索引(即下标)
    validEcacheLsit = nonzero(oS.eCache[:, 0].A)[0]
    if (len(validEcacheLsit) > 1):
        for k in validEcacheLsit:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


#更新误差
def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


#完整版Platt SMO内循环
#输出：是否在数据结构中成功更新alpha，成功返回1，不成功返回0
def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or\
        ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        #公式见《统计学习方法》或西瓜书
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L == H")
            return 0
        #见《统计学习方法》公式7.107
        # eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T -\
        #       oS.X[j, :] * oS.X[j, :].T
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
        if eta >= 0:
            print("eta >= 0")
            return 0
        #见《统计学习方法》公式7.106
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)
        #alphaJ优化幅度太小，返回0
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print("j not moving enough")
            return 0
        #见《统计学习方法》公式7.109
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j]) 
        updateEk(oS, i)
        #见《统计学习方法》公式7.115,7.116
        # b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[i, :].T - \
        #      oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[i, :] * oS.X[j, :].T
        # b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[j, :].T - \
        #      oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].T
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - \
             oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - \
             oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[j, j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


# 功能：完整版Platt SMO外循环
# 输入：特征矩阵dataMatIn，标签向量classLabels，常数C，容错率toler，最大迭代次数maxIter
# 输出：超平面位移项b，拉格朗日乘子alphas
def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler, kTup)
    iter = 0
    entireSet = True #是否遍历整个数据集
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or entireSet):
        alphaPairsChanged = 0
        if entireSet: #判断1，遍历整个集合
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
                print("fullSet, iter: %d, i: %d, pairs changed: %d" % (iter, i, alphaPairsChanged))
            iter += 1
        else:#遍历非边界值
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("non-bound, iter: %d, i: %d, pairs changed: %d" % (iter, i, alphaPairsChanged))
            iter += 1
        # 执行判断1时，如果entireSet == True，表示遍历整个集合；alphaPairsChanged == 0，表示未对任意alpha进行修改
        # 执行判断1时，第一次迭代遍历整个集合，之后就只遍历非边界值；如果遍历非边界值发现没有任意alpha对进行修改，则遍历整个集合
        # 如果迭代次数超过指定最高值 或遍历整个集合后 alphaPairsChanged仍然等于0（表明优化完毕），结束迭代
        if entireSet:
            entireSet = False
        elif (alphaPairsChanged == 0):
            entireSet = True
        print("iteration number: %d" % iter)
    return oS.b, oS.alphas


def calcW(alphas, dataArr, classLabels):
    X = mat(dataArr)
    labelMat = mat(classLabels).T
    m, n = shape(X)
    W = zeros((n, 1))
    for i in range(m):
        W += multiply(alphas[i] * labelMat[i], X[i, :].T)
    return W


def testRBF(k1=1.3):
    dataArr, labelArr = loadData("testSetRBF.txt")
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svIndex = nonzero(alphas.A>0)[0]
    SVs = dataMat[svIndex]
    labelSV = labelMat[svIndex]
    print("there are %d Support Vectors" % shape(SVs)[0])
    m, n = shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(SVs, dataMat[i, :], ('rbf', k1))
        predict = kernelEval.T * multiply(labelSV, alphas[svIndex]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print("the training error rate is: %f" % (float(errorCount) / m))

    dataArr, labelArr = loadData("testSetRBF2.txt")
    errorCount = 0
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m, n = shape(dataMat)
    for i in range(m):
        kernelEval = kernelTrans(SVs, dataMat[i, :], ('rbf', k1))
        predict = kernelEval.T * multiply(labelSV, alphas[svIndex]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print("the test error rate is: %f" % (float(errorCount) / m))


def img2vector(filename):
    '''
    将32*32的二进制图像矩阵转换为1*1024的向量
    '''
    returnVect = zeros((1,1024))
    with open(filename) as fr:
        for i in range(32):
            lineStr = fr.readline()
            for j in range(32):
                returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect


def loadImages(dirName):
    '''
    处理手写数字文件，返回特征矩阵featureMat和类别标签hwlabels
    '''
    from os import listdir
    hwLabels = []
    trainFileList = listdir(dirName)
    m = len(trainFileList)
    featureMat = zeros((m, 1024)) #每个手写数字样本为1*1024的行向量
    for i in range(m):
        fileNameStr = trainFileList[i] #获取文件名
        fileStr = fileNameStr.split('.')[0] #文件名前缀
        classNameStr = int(fileStr.split('_')[0]) #样本代表的手写数字
        # 将手写数字9划分为"-1"类，其余数字划分为"1"类，共两类
        if classNameStr == 9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        featureMat[i, :] = img2vector(("%s/%s") % (dirName, fileNameStr))
    return featureMat, hwLabels


def testDigits(kTup=('rbf', 10)):
    dataArr, labelArr = loadImages("F:/codes/python/Machine-Learning-In-Action-python3/kNN/digits/trainingDigits")
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svIndex = nonzero(alphas > 0)[0] #支持向量索引
    SVs = dataMat[svIndex] # 支持向量
    SVLabel = labelMat[svIndex] # 支持向量标签
    print("there are %d Support Vectors" % shape(SVs)[0])
    m, n = shape(dataMat)
    errorCount = 0

    #训练数据错误率
    for i in range(m):
        kernelEval = kernelTrans(SVs, dataMat[i, :], kTup)
        predict = kernelEval.T * multiply(SVLabel, alphas[svIndex]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print("the training error rate is: %f" % (float(errorCount) / m))

    #测试数据错误率
    dataArr, labelArr = loadImages("F:/codes/python/Machine-Learning-In-Action-python3/kNN/digits/testDigits")
    errorCount = 0
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m, n = shape(dataMat)
    for i in range(m):
        kernelEval = kernelTrans(SVs, dataMat[i, :], kTup)
        predict = kernelEval.T * multiply(SVLabel, alphas[svIndex]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print("the test error rate is: %f" % (float(errorCount) / m))



if __name__ == "__main__":
    start = time.time()
    testDigits(('lin', 0))
    end = time.time()
    print("\n\ntiming: %f s" % (end - start))