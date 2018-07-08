# coding=utf-8
#####################lesson 3#####################
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sympy.functions.elementary.trigonometric import csc

x=np.linspace(-1,1,50)
y=2*x+1
plt.plot(x,y)
plt.show()

y=x**2
plt.plot(x,y)
plt.show()

#####################lesson 4 figure#####################
####################linewidth linestyle color num figsize#####################
#################### figure就是最外面的窗口 #####################
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

x=np.linspace(-3,3,50)
y1=x*2+1
y2=x**2
#####两个figure() 指的是两个figure窗口
plt.figure()
plt.plot(x,y1)

plt.figure(num=3,figsize=(8,5))
plt.plot(x,y2)
plt.show()
#####两个figure放在同一个窗口
plt.figure()
plt.plot(x,y1)
plt.plot(x,y2,color="red",linewidth=1.0,linestyle="--")
plt.show()
#####################lesson 5 axis 坐标轴设置1#####################
####################xlim ylim xticks yticks 坐标轴自定义#####################
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
x=np.linspace(-3,3,50)
y1=x*2+1
y2=x**2

plt.figure()
plt.plot(x,y1)
plt.plot(x,y2,color="red",linewidth=1.0,linestyle="--")

plt.xlim((-1,1))
plt.ylim((-2,3))
plt.xlabel("T am XLable")
plt.ylabel("T am YLable")
new_ticks = np.linspace(-1,2,5)
plt.xticks(new_ticks)
plt.yticks([-2,-1.8,-1,1.22,3,],[r"$really\ bad$",r"$bad$",r"$noraml\ \alpha$",r"$good$",r"$really\ good$"])
plt.show()
#####################lesson 6 axis 坐标轴设置2#####################
####################gca ax#####################
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

x=np.linspace(-3,3,50)
y1=x*2+1
y2=x**2

plt.figure()
plt.plot(x,y1)
plt.plot(x,y2,color="red",linewidth=1.0,linestyle="--")

#gca:get current axis

ax = plt.gca()
ax.spines["right"].set_color("none")
ax.spines["top"].set_color("none")
ax.xaxis.set_ticks_position("bottom")
ax.yaxis.set_ticks_position("left")
ax.spines["bottom"].set_position(("data",0)) #outward（没有查到） axes(定位到百分之多少的位置)
ax.spines["left"].set_position(("data",0))

plt.show()
#####################lesson 7 axis 坐标轴设置2#####################
#################### legend l1, l2, loc handles labels#####################
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
x=np.linspace(-3,3,50)
y1=x*2+1
y2=x**2

plt.figure()
plt.xlim((-1,1))
plt.ylim((-2,3))
plt.xlabel("T am XLable")
plt.ylabel("T am YLable")
new_ticks = np.linspace(-1,2,5)
plt.xticks(new_ticks)
plt.yticks([-2,-1.8,-1,1.22,3,],[r"$really\ bad$",r"$bad$",r"$noraml\ \alpha$",r"$good$",r"$really\ good$"])

# plt.plot(x,y1,label="up")
# plt.plot(x,y2,color="red",linewidth=1.0,linestyle="--",label="down")
# plt.legend()

l1,=plt.plot(x,y1,label="up")
l2,=plt.plot(x,y2,color="red",linewidth=1.0,linestyle="--",label="down")
plt.legend(handles=[l1,l2,],labels=["aaaa","bbbb"],loc="best")
plt.show()
#####################lesson 8 annotation#####################
#################### #####################
#####################lesson 9 axis tick 坐标轴刻度#####################
#################### #####################
#####################lesson 10 scatter 散点图数据#####################
#################### #####################
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

n =1024
X = np.random.normal(0,1,n)
Y = np.random.normal(0,1,n)
T = np.arctan2(X,Y)
plt.scatter(X,Y,s=75,c=T,alpha=0.75)

plt.xlim((-1.5,1.5))
plt.ylim((-1.5,1.5))
plt.xticks(())
plt.yticks(())
plt.show()
#####################lesson 11 bar 柱状图#####################
#################### facecolor edgecolor zip uniform #####################
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

n =12
X =np.arange(n)
Y1=(1-X/float(n))*np.random.uniform(0.5,1.0,n)
Y2=(1-X/float(n))*np.random.uniform(0.5,1.0,n)

plt.bar(X,Y1,facecolor="#9999ff",edgecolor="white")
for x,y in zip(X,Y1) :
	plt.text(x+0.04,y+0.05,"%.2f" %y,ha="center",va="bottom")

plt.bar(X,-Y2,facecolor="#ff9999",edgecolor="white")
for x,y in (X,Y2) :
	plt.text(x+0.04,-y-0.2,"%.2f" %y,ha="center",va="bottom")
plt.xlim(-1,n)
plt.ylim(-1.25,1.25)

plt.show()
#####################lesson 12 contour 等高线图#####################
####################   #####################
#####################lesson 13 image 图片#####################
####################   #####################
#####################lesson 14 3D plot 3D数据#####################
####################  rstride=行跨,cstride=列跨 #####################
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)

X = np.arange(-4,4,0.25)
Y = np.arange(-4,4,0.25)
X ,Y =np.meshgrid(X,Y)
R =np.sqrt(X**2+Y**2)
Z = np.sin(R)

ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap=plt.get_cmap("rainbow"))
ax.contour(X,Y,Z,zdir="z",offset=-2,cmap="rainbow")
ax.set_zlim(-2,3)

plt.show()
#####################lesson 15 subplot 多个显示#####################
####################   #####################
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.subplot(2,2,1)
plt.plot([0,1],[0,1])
plt.subplot(2,2,2)
plt.plot([0,1],[0,1])
plt.xlabel("picture 1")
plt.ylabel("picture 2")
plt.subplot(2,2,3)
plt.plot([0,1],[0,1])
plt.subplot(2,2,4)
plt.plot([0,1],[0,1])

plt.show()


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# fig,ax1 = plt.subplot(2,1,1) fig与下面等价
plt.subplot(2,1,1)
plt.plot([0,1],[0,1])
plt.subplot(2,3,4)#再次重新划分fig
plt.plot([0,1],[0,1])
plt.xlabel("picture 1")
plt.ylabel("picture 2")
plt.subplot(235)
plt.plot([0,1],[0,1])
plt.subplot(236)
plt.plot([0,1],[0,1])

plt.show()
#####################lesson 16 subplot in grid 多个显示#####################
####################   #####################
#####################lesson 17 plot in plot 画中画#####################
####################   #####################
#####################lesson 18 secondary axis 次坐标#####################
####################   #####################
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

x = np.arange(0,10,0.1)
y1 = 0.05*x**2
y2 = 1/(x+0.01)*10+10

fig,ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(x,y1,"g-")
ax2.plot(x,y2,"b--")

ax1.set_xlabel("X data")
ax1.set_ylabel("Y1",color="g")
ax2.set_ylabel("Y2",color="b")

plt.show()

#####################lesson 19 animation 动画#####################
####################   #####################










