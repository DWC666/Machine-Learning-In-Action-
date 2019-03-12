import numpy as np
import operator
from os import listdir

def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    dataSize = dataSet.shape[0]
    #求目标样本与各训练样本的差值
    diffMat = np.tile(inX, (dataSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    #获取排序索引值
    sortedDisIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDisIndicies[i]]
        #统计前k个近邻中各类的数量
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def autoNorm(dataSet):
    '''
    归一化特征值矩阵
    '''
    #参数0使得函数从每列中选取最小（大）值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m,1))
    normDataSet = normDataSet / np.tile(ranges, (m,1))

    return normDataSet, ranges, minVals

def img2vector(filename):
    '''
    将32*32的二进制图像矩阵转换为1*1024的向量
    '''
    returnVect = np.zeros((1,1024))
    with open(filename) as fr:
        for i in range(32):
            lineStr = fr.readline()
            for j in range(32):
                returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    '''
    手写数字识别
    '''
    hwLabels = []
    trainingFileList = listdir("F:\\codes\\python\\machineLearningInAction\\kNN\\digits\\trainingDigits")
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('F:\\codes\\python\\machineLearningInAction\\kNN\\digits\\trainingDigits\\%s' % fileNameStr)
    testFileList = listdir("F:\\codes\\python\\machineLearningInAction\\kNN\\digits\\testDigits")
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector("F:\\codes\\python\\machineLearningInAction\\kNN\\digits\\testDigits\\%s" % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if(classifierResult != classNumStr):
            errorCount += 1.0
    print("the total number of errors is: %d" % errorCount)
    print("the total error rate is: %f" % (errorCount/mTest))


if __name__ == "__main__":
    handwritingClassTest()
