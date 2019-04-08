from numpy import *
import matplotlib.pyplot as plt

def loadSampleData():
    dataMat = matrix([[1., 2.1], [2., 1.1], [1.3, 1.], [1., 1.], [2., 1.]])
    classMat = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classMat


def loadDataSet(fileName):
    numFeature = len(open(fileName).readline().split('\t'))
    dataMat = []
    labelMat = []
    with open(fileName) as fr:
        for line in fr.readlines():
            lineArr = []
            curLine = line.strip().split('\t')
            for i in range(numFeature -1):
                lineArr.append(float(curLine[i]))
            dataMat.append(lineArr)
            #将标签0转换成-1，便于后面取符号计算
            if float(curLine[-1]) == 0:
                labelMat.append(-1.0)
            else:
                labelMat.append(float(curLine[-1]))
            # labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

def stumpClassify(dataMat, dimen, threshVal, threshIneq):
    '''
    通过阈值比较对样本进行分类
    @dataMat: 特征矩阵
    @dimen: 维度
    @threshVal: 阈值
    @threshIneq: 不等号
    @return retArray: 分类结果
    '''
    retArray = ones((shape(dataMat)[0], 1))
    if threshIneq == "lt":
        retArray[dataMat[:, dimen] <= threshVal] = -1
    else:
        retArray[dataMat[:, dimen] > threshVal] = -1
    return retArray


def buildStump(dataArr, classLabels, D):
    '''
    返回最佳单层决策树
    @dataArr: 特征矩阵
    @classLabels: 标签
    @D: 权重向量
    '''
    dataMat = mat(dataArr)
    labelMat = mat(classLabels).T
    m, n = shape(dataMat)
    #特征值上的遍历步数
    numSteps = 10.0
    #字典存储最佳决策树的相关参数
    bestStump = {}
    bestClasEst = mat(zeros((m, 1)))
    minError = inf #init error sum, to +infinity
    for i in range(n):
        rangeMin = dataMat[:, i].min()
        rangeMax = dataMat[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps)+1): #loop over all range in current dimension
            for inequal in ["lt", "gt"]: #go over less than and greater than
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMat, i, threshVal, inequal)
                errorArr = mat(ones((m, 1)))
                # 预测标签和真实标签相等则相应位置为0， 不等则为1
                errorArr[predictedVals == labelMat] = 0
                # print("predictedVals: ", predictedVals.T, "errorArr: ", errorArr.T)
                weightedError = D.T * errorArr #calc total error multiplied by D
                # print("split: dim %d, thresh %.2f, thresh inequal: %s, weightedError: %.3f" % (i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump["dim"] = i
                    bestStump["thresh"] = threshVal
                    bestStump["ineq"] = inequal
    
    return bestStump, minError, bestClasEst


def adaBoostTrain(dataArr, classLabels, numIt=50):
    '''
    利用单层决策树构建adaBoost算法
    @dataArr: 特征数组
    @classLabels: 标签
    @numIt: 迭代次数，同时也是弱分类器个数的最大值
    '''
    weakClassifier = [] #存储弱分类器
    m = shape(dataArr)[0]
    D = mat(ones((m, 1))/m) #样本权重向量，初始化为相等
    aggClassEst = mat(zeros((m, 1))) #列向量， 记录每个样本想的类别估计累计值

    for _ in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        # print("error:", error)
        # print("D: ", D.T)
        #弱分类器权重。calc alpha, throw in max(error,eps) to account for error=0
        alpha = float(0.5 * log((1.0-error) / max(error, 1e-16))) 
        bestStump["alpha"] = alpha
        # print("alpha: ", alpha)
        weakClassifier.append(bestStump)
        # print("classEst: ", classEst.T)
        # 增加错分样本的权重，减少正确划分的样本的权重
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst) #exponent for D calc, getting messy
        D = multiply(D, exp(expon)) #Calc New D, element-wise 
        D = D / D.sum() #使其符合权重性质，即相加和为1
        
        aggClassEst += alpha * classEst
        # print("aggClassEst: ", aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        # print("aggErrors: ", aggErrors.T)
        errorRate = aggErrors.sum() / m
        # print("errorRate: ", errorRate, "\n\n")
        #如果错误率为0，则退出
        if errorRate == 0.0:
            break
    return weakClassifier, aggClassEst


def adaClassify(dataToClass, classifierArr):
    dataMat = mat(dataToClass)
    m = shape(dataMat)[0]
    aggClassEst = mat(zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMat, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
    print(aggClassEst.T)
    return sign(aggClassEst)


def plotROC(predStrengths, classLabels):
    '''
    绘制ROC曲线：
    将分类样例的预测强度（即adaBoost算法的预测概率值）升序排列，取其索引值，
    依次遍历每个索引，以每个索引对应的预测强度为阈值，大于等于该阈值则划分为正立，反之划分为负例。
    Parameters:
        predStrengths：样例的预测强度，必须为横向量
        classLabels：样例的真实标签
    '''
    #光标位置，初始化为(1.0, 1.0)，即右上角
    cur = (1.0, 1.0)
    #累加所有点的Y轴长度，方便后面计算AUC
    ySum = 0.0
    #正例数量，即TP+FN
    numPosClas = sum(array(classLabels) == 1.0)
    #Y轴上的步长，1/(TP+FN)
    yStep = 1 / float(numPosClas)
    #X轴上的步长，1/(FP+TN)
    xStep = 1 / float(len(classLabels) - numPosClas)
    #预测强度升序排列，取其索引值
    sortedIndicies = predStrengths.argsort()

    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)

    for index in sortedIndicies.tolist()[0]:
        #每得到一个标签为1.0的类，就沿Y轴方向下降一个步长
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        #对于其他标签，则在X轴左移一个步长
        else:
            delX = xStep
            delY = 0
            #累加矩形的高
            ySum += cur[1]
        #连接上一个和下一个点
        ax.plot([cur[0], cur[0]-delX], [cur[1], cur[1] - delY], c='b')
        #绘制左下角到右上角的对角虚线
        cur = (cur[0]-delX, cur[1]-delY)
        print(index, cur)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    ax.axis([0, 1, 0, 1])
    #所有小矩形的高的和(ySum)乘以宽(xStep)即为AUC
    AUC = ySum * xStep
    print("the Area Under the Curve is: ", AUC)
    plt.show()
    



if __name__ == "__main__":
    dataArr, classLabels = loadDataSet("horseColicTraining.txt")
    # print(classLabels)
    # testArr, testlabels = loadDataSet("horseColicTest.txt")

    classifierArr, predStrengths = adaBoostTrain(dataArr, classLabels)
    # print(predStrengths.T)
     
    # aggClassEst = adaClassify(testArr, classifierArr)
    # print(aggClassEst.T)
    # errArr = mat(ones((67, 1)))
    # errorCount = errArr[aggClassEst != mat(testlabels).T].sum()
    # errorRate = errorCount / shape(errArr)[0]
    # print("classifier number: ", len(classifierArr))
    # print("error Rate: ", errorRate)

    plotROC(predStrengths.T, classLabels)