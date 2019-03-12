import numpy as np

def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'], 
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'], 
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'], 
        ['mr', 'licks', 'ate', 'steak', 'how', 'to', 'stop', 'him'], 
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]

    classVec = [0, 1, 0, 1, 0, 1]#1代表侮辱性文字， 0代表正常文字
    return postingList, classVec

'''创建单词表，先用set去重，然后再以list的形式返回'''
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


'''返回一个01列表，表示单词表中的某个单词是否出现在inputSet中，称为词集模型'''
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my vocabulary!" % word)
    return returnVec

'''词袋模型：计算单词表中每个单词在inputSet中出现次数'''
def bagOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print("the word: %s is not in my vocabulary!" % word)
    return returnVec

'''求出P(c1)、P(w|c1)、P(w|c0).   当P(c1)求出时，P(c0) = 1-P(c1)'''
'''P(w|c1)是一个列表，等于[P(w1|c1),P(w2|c1),P(w3|c1)……]; P(w|c0)也是一个列表；P(c1)为一个实数'''
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = np.sum(trainCategory) / float(numTrainDocs)  #P(c1)
    '''
         以下是计算P(w|c1)、P(w|c0)的初始化阶段。
         在计算P(w1|c1)*P(w2|c0)*……时，如果其中一个为0，那么结果就为0，为了降低这种影响，
         可使用拉普拉斯平滑：分子初始化为1，分母初始化为2（有几个分类就初始化为几。这里有两个分类，所以初始化为2）
    '''
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += np.sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += np.sum(trainMatrix[i])
    '''
          以下是计算P(w|c1)、P(w|c0)的最终部分。
          在计算P(w1|c1)*P(w2|c0)*……时，由于太多很小的数相乘，很容易造成下溢，当四舍五入时结果就很可能为0，
          解决办法是对乘积取对数，即：ln(ab) = lna+lnb 把“小数相乘”转化为小数相加，避免了下溢。
          由于x与lnx在x>0处有相同的增减性，两者的极值不相同，但极值点是相同的，不会影响最终的分类。
    '''
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    '''
        vec2Classify * p1Vec是什么意思？
        可知p1Vec为训练数据集中的P(w|c1)，它是一个列表（对应着单词表）。在计算P(w|c1)时，每一个单词的概率即P(wi|c1)都计算了，
        但对于输入数据而言，有些单词出现了，有些单词没有出现，而我们只需要计算出现了的单词，所以要乘上系数vec2Classify
      
        注意：对数加减法对应于原式的乘除法
        sum(vec2Classify * p1Vec)计算的是P(w|c1)，log(pClass1)计算的是P(c1)。
        按照贝叶斯公式，理应还要减去P(w)的那一部分。但由于p1、p0都需要减去这一部分的值，
        故只需要比较p1、p0的大小而不必求出具体的值。
      '''
    p1 = np.sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = np.sum(vec2Classify * p0Vec) + np.log(1-pClass1)
    if p1 > p0:
        return 1
    return 0

def testingNB():
    listOPsts, listClasses = loadDataSet()
    vocabList = createVocabList(listOPsts)
    trainMat = []
    for postinDoc in listOPsts:
        trainMat.append(setOfWords2Vec(vocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(vocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(vocabList, testEntry))
    print(testEntry, 'classified as: ',classifyNB(thisDoc, p0V, p1V, pAb))

def textParse(bigString):  # 提取、处理、过滤单词
    import re
    # '\w*' 会匹配0个或多个规则，split会将字符串分割成单个字符【python3.5+】; 这里使用 \W 或者 \W+ 都可以将字符数字串分割开
    # 产生的空字符将会在后面的列表推导式中过滤掉
    listOfTokens = re.split(r'\W+', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        #读取每个垃圾邮件，并字符串转换成字符串列表
        wordList = textParse(open('F:\\codes\\python\\machineLearningInAction\\NaiveBayes\\email\\spam\\%d.txt' % i, 'r').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1) #标记垃圾邮件，1表示垃圾邮件
        #读取每个正常邮件
        wordList = textParse(open('F:\\codes\\python\\machineLearningInAction\\NaiveBayes\\email\\ham\\%d.txt' % i, 'r').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50))
    testSet = []
    #在50个邮件中随机选取10个作为测试邮件，剩余40个作为训练集
    for i in range(10):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []; trainClass = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClass.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClass))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print("the error rate is: ", float(errorCount)/len(testSet))
for i in range(10):
    spamTest()
