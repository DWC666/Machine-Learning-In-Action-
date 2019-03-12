#coding:utf-8
import matplotlib.pyplot as plt
#为了在图中显示中文
from pylab import *  
mpl.rcParams['font.sans-serif'] = ['SimHei']  

# boxstyle为文本框的类型，sawtooth是锯齿形，fc是边框线粗细
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def plotMidText(cntrPt, parentPt, txtString):
        #在父子节点中间填充文本信息
        xMid = (parentPt[0] - cntrPt[0])/2.0 + cntrPt[0]
        yMid = (parentPt[1] - cntrPt[1])/2.0 + cntrPt[1]
        createPlot.ax1.text(xMid, yMid, txtString)

def plotTree(myTree, parentPt, nodeTxt):
        numLeafs = getNumLeafs(myTree)
        depth = getTreeDepth(myTree)
        firstStr = list(myTree.keys())[0]
        cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
        plotMidText(cntrPt, parentPt, nodeTxt)
        plotNode(firstStr, cntrPt, parentPt, decisionNode)
        secondDict = myTree[firstStr]
        plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
        for key in secondDict.keys():
                if type(secondDict[key]).__name__ == 'dict':      #test to see if the nodes are dictonaires, if not they are leaf nodes   
                        plotTree(secondDict[key], cntrPt, str(key))        #recursion
                else:   #it's a leaf node print the leaf node

                        plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
                        plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
                        plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
        plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
#if you do get a dictonary you know it's a tree, and the first element will be another dict



def plotNode(nodeText, centerPt, partentPt, nodeType):
    # annotate是关于一个数据点的文本  
    # nodeTxt为要显示的文本，centerPt为文本的中心点，箭头所在的点，parentPt为指向文本的点 
    createPlot.ax1.annotate(nodeText, xy=partentPt, xycoords='axes fraction', xytext=centerPt,\
        textcoords='axes fraction', va='center', ha='center', bbox=nodeType, arrowprops=arrow_args)

def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()#画布清空
    # 函数属性createPlot.ax1为全局变量，绘制图像的句柄，subplot为定义了一个绘图，
    #111表示figure中的图有1行1列，即1个，最后的1代表第一个图 
    # frameon表示是否绘制坐标轴矩形
    axprops = dict(xticks=[], yticks=[]) 
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()
    

def getNumLeafs(myTree):
        numLeafs = 0
        firstStr = list(myTree.keys())[0]
        secondDict = myTree[firstStr]
        for key in secondDict.keys():
                if type(secondDict[key]).__name__ == 'dict':
                        numLeafs += getNumLeafs(secondDict[key])
                else:
                        numLeafs += 1
        return numLeafs

def getTreeDepth(myTree):
        maxDepth = 0
        firstStr = list(myTree.keys())[0]
        secondDict = myTree[firstStr]
        for key in secondDict.keys():
                if type(secondDict[key]).__name__ == 'dict':
                        thisDepth = 1 + getTreeDepth(secondDict[key])
                else:
                        thisDepth = 1
                if thisDepth > maxDepth:
                        maxDepth = thisDepth
        return maxDepth

