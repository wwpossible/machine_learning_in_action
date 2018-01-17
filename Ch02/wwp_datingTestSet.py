import numpy as np
import sys
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from imp import reload
sys.path.append(r"C:\Python34\code\machinelearninginaction\Ch02")
import kNN

reload(kNN)
'''
datingDataMat, datingLabels = kNN.file2matrix(datingTestSet2.txt)
tt=15.0*np.array(datingLabels)
fig = plt.figure()
ax1 = fig.add_subplot(311)#第一幅图
#散点图,X:每年获得的飞行常客里程数, Y:玩视频游戏所耗时间百分比，两个10*np.array(datingLabels)是scatter中的用法，用数字标识XY值的属性类别
ax1.scatter(datingDataMat[:,0], datingDataMat[:,1], tt, tt)

ax2 = fig.add_subplot(312)#第二幅图
#散点图,X:每年获得的飞行常客里程数, Y:每周所消费的冰淇淋公升数，两个10*np.array(datingLabels)是scatter中的用法，用数字标识XY值的属性类别
ax2.scatter(datingDataMat[:,0], datingDataMat[:,2], tt, tt)

ax3 = fig.add_subplot(313)#第三幅图
#散点图,X:玩视频游戏所耗时间百分比, Y:每周所消费的冰淇淋公升数，两个10*np.array(datingLabels)是scatter中的用法，用数字标识XY值的属性类别
ax3.scatter(datingDataMat[:,1], datingDataMat[:,2], tt, tt)

plt.show()
'''

#kNN.datingClassTest()
kNN.classifyPerson()