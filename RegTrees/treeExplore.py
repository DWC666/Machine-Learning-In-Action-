'''
构建一个回归树或模型树的GUI，支持修改参数和重新绘制等操作
'''
from tkinter import *

import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from numpy import *

import regTrees
# Agg是一个C++的库，可以从图像创建位图（像素图）
# TkAgg可以在所选的GUI框架上调用Agg，把Agg呈现在画布上
matplotlib.use('TkAgg')


def reDraw(tolS, tolN):
    '''
    函数说明：在数据点上绘制回归或模型树
    '''
    reDraw.f.clf()  # clear the figure
    reDraw.a = reDraw.f.add_subplot(111)
    # 如果复选框选中，则构建模型树
    if chkBtnVar.get():
        if tolN < 2:
            tolN = 2
        myTree = regTrees.createTree(reDraw.rawDat, regTrees.modelLeaf,
                                     regTrees.modelErr, (tolS, tolN))
        yHat = regTrees.createForeCast(myTree, reDraw.testDat, regTrees.modelTreeEval)
    # 如果复选框未选中，则构建回归树
    else:
        myTree = regTrees.createTree(reDraw.rawDat, ops=(tolS,tolN))
        yHat = regTrees.createForeCast(myTree, reDraw.testDat)

    reDraw.a.scatter(list(reDraw.rawDat[:,0]), list(reDraw.rawDat[:,1]), s=5) #use scatter for data set
    reDraw.a.plot(reDraw.testDat, yHat, linewidth=2.0) #use plot for yHat
    reDraw.canvas.draw()


def getInputs():
    '''
    函数说明：返回输入框中的tolS和tolN
    '''
    try:
        # 期望输入浮点数
        tolS = float(tolSentry.get())
    except: 
        tolS = 1.0 
        print("enter Float for tolS")
        # 清空输入框
        tolSentry.delete(0, END)
        # 输入框恢复默认值 1.0
        tolSentry.insert(0,'1.0')

    try:
        # 期望输入整数
        tolN = int(tolNentry.get())
    except:
        tolN = 10 
        print("enter Integer for tolN")
        # 清空输入框
        tolNentry.delete(0, END)
        # 输入框恢复默认值 10
        tolNentry.insert(0, '10')
    
    return tolS,tolN

def drawNewTree():
    '''
    点击reDraw按钮时会调用该函数
    '''
    tolS, tolN = getInputs()
    reDraw(tolS, tolN)


root = Tk()

reDraw.f = Figure(figsize=(5,4), dpi=100) #create figure
# 通过渲染器创建画布组件
reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master=root)
# 显示画布
reDraw.canvas.draw()
reDraw.canvas.get_tk_widget().grid(row=0, columnspan=3)

# 创建tolN标签，并设置布局
Label(root, text="tolN").grid(row=1, column=0)
# 创建输入框
tolNentry = Entry(root)
tolNentry.grid(row=1, column=1)
# 设置输入框默认值为10
tolNentry.insert(0, '10')
# 创建tolS标签，并设置布局
Label(root, text="tolS").grid(row=2, column=0)
tolSentry = Entry(root)
tolSentry.grid(row=2, column=1)
tolSentry.insert(0, '1.0')
# 创建按钮ReDraw，设置调用函数为drawNewTree
Button(root, text="ReDraw", command=drawNewTree).grid(row=1, column=2, rowspan=3)
# 复选框按钮整数值对象，判断复选框是否选中
chkBtnVar = IntVar()
# 创建复选框按钮
chkBtn = Checkbutton(root, text="Model Tree", variable = chkBtnVar)
chkBtn.grid(row=3, column=0, columnspan=2)

# 创建函数reDraw()下的全局变量，存储原始数据
reDraw.rawDat = mat(regTrees.loadDataSet('sine.txt'))
# 测试数据
reDraw.testDat = arange(min(reDraw.rawDat[:,0]),max(reDraw.rawDat[:,0]),0.01)
# 运行函数，否则初始运行不会绘制图像
reDraw(1.0, 10)
# print(reDraw.testDat)
root.mainloop()