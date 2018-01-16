from numpy import *
import sys
import matplotlib
import matplotlib.pyplot as plt
from imp import reload
sys.path.append(r"C:\Python34\code\machinelearninginaction\Ch02")
import kNN

reload(kNN)

df = kNN.createDataSet()
inputt = array([0.7,0.8])
K = 3
output = kNN.classify(inputt,df,K)
print("测试数据为:",inputt,"分类结果为：",output)

fig = plt.figure(figsize=(6,6))#XY轴具有相同的刻度和比例
ax = fig.add_subplot(1, 1, 1)
plt.plot(df['x'], df['y'], 'ro')#画图
plt.plot(inputt[0], inputt[1], 'go')
count = 0
##添加标注
for label in df.index:
    ax.annotate(label,
                xy=df.values[count],
                xytext=(df.values[count][0]+0.1, df.values[count][1]+0.05),
                arrowprops=(dict(facecolor='b', width=0.05, shrink=0.05, headwidth=1, connectionstyle="arc3")))
    count+=1
##添加input的标注
ax.annotate(output,
                xy=inputt,
                xytext=(inputt[0]+0.1, inputt[1]+0.05),
                arrowprops=(dict(facecolor='b', width=0.05, shrink=0.05, headwidth=1, connectionstyle="arc3")))

plt.grid(True)#产生网格
plt.show()#显示图像
