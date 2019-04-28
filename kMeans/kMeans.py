from numpy import *


def loadDataSet(fileName):
    '''
    加载数据
    '''
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat


def distEclud(vecA, vecB):
    '''
    计算两个向量的欧式距离
    '''
    return sqrt(sum(power(vecA - vecB, 2)))


def randCent(dataset, k):
    '''
    生成k个随机质心
    '''
    n = shape(dataset)[1]

    centroids = mat(zeros((k, n)))
    for j in range(n):
        # j列的最小值
        minJ = min(dataset[:, j])
        # j列的最大值与最小值之差
        rangeJ = float(max(dataset[:, j]) - minJ)
        # 对于j列,生成边界内的 k 个随机数
        centroids[:, j] = mat(minJ + rangeJ * random.rand(k, 1))
    return centroids


def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    '''
    函数说明：K均值聚类
    @Param:
        dataSet -> 数据矩阵（np.matrix类型）
        k -> 簇的数目
        distMeas -> 距离计算函数
        createCent -> 创建初始质心的函数
    '''
    m = shape(dataSet)[0]
    # 用来存储每个数据点的簇分配结果的矩阵
    # 矩阵包含两列：第一列记录簇索引值，第二列存储与质心的误差
    clusterAssment = mat(zeros((m, 2)))
    # 生成k个初始质心
    centroids = createCent(dataSet, k)
    # 标志变量
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            # 如果不相等，则更新标志变量
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist**2
        # print(centroids)
        for cent in range(k):
            # 筛选出簇中所有数据点
            ptsInclust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]
            # 用簇中数据的均值来更新质心的取值
            centroids[cent, :] = mean(ptsInclust, axis=0)
    return centroids, clusterAssment


def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    # 用每列的均值生成一个初始质心
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    # 用一个列表来存储每个簇的质心
    centList = [centroid0]
    for j in range(m):
        # 用SSE（误差平方和）来描述一个数据点到质心的距离，以及聚类效果
        clusterAssment[j, 1] = distMeas(mat(centroid0), dataSet[j, :])**2
    # 直到得到想要的簇数k才停止循环
    while len(centList) < k:
        lowestSSE = inf
        # 遍历每个簇来寻找二分后最大程度降低SSE的簇
        for i in range(len(centList)):
            # 当前簇中的数据点
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]
            # 对当前簇中的数据点进行二分，得到两个新簇，分别为0和1
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:, 1])
            sseNoSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])
            print("sseSplit: %f,  sseNoSplit: %f" % (sseSplit, sseNoSplit))
            # 新的两个簇的误差与剩余数据点的误差之和作为本次划分的误差
            # 如果本次划分误差小于当前最小误差，则保留相关数据
            if (sseSplit + sseNoSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNoSplit
        # 对最终确定的要划分的簇生成的簇中的数据的簇编号进行修改（原编号为0和1，会造成混乱）
        # 将新簇中标记为1的簇改为新增的簇编号，即len(centList)
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList)
        # 将新簇中标记为0的簇改为生成该簇的原簇编号，即bestCentToSplit
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        print('the bestCentToSplit is: ',bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))
        # 用新生成的第一个质心替换原质心
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]#replace a centroid with two best centroids 
        # 将新生成的第二个质心加入列表
        centList.append(bestNewCents[1,:].tolist()[0])
        # 更新簇的分配结果
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:] = bestClustAss#reassign new clusters, and SSE
    return mat(centList), clusterAssment



if __name__ == "__main__":
    data = loadDataSet('testSet2.txt')
    dataMat = mat(data)
    centroids, clusterAssment = biKmeans(dataMat, 3)
    print(centroids)
   