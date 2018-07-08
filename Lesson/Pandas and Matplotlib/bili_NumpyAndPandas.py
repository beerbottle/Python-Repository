# coding=utf-8
###################################### Lesson1 ######################################
import numpy as np
array = np.array([[1,2,3],[2,3,4]])
array
print(array)

print("number of dim:",array.ndim)
print("shape of dim:",array.shape)
print("size of dim:",array.size)
###################################### Lesson2 ######################################
import numpy as np

a = np.array([[2,3,4,5],
             [2,3,2,54]])
print(a)
a = np.array([2,3,4,5],dtype=np.int)
a = np.array([2,3,4,5],dtype=np.int64)
a = np.array([2,3,4,5],dtype=np.int32)
a = np.array([2,3,4,5],dtype=np.float64)
a = np.array([2,3,4,5],dtype=np.float32)
print(a.dtype)

b = np.zeros((3,4))
c = np.ones((3,4))
d = np.empty((3,4))

e = np.arange(1,10,2) #和python中的range差不多
f = np.arange(12).reshape((3,4))
g = np.linspace(1,20,3).reshape((3,4))
###################################### Lesson3 ######################################
import numpy as np

a = np.array([10,20,30,40])
b = np.arange(4)

print(a+b)
print(a-b)
print(a*b)
print(b**2)
print(10*np.sin(a))
print(b < 3)

c = np.array([[1,2],[3,4]])
d = np.arange(4).reshape((2,2))
print(c*d)
print(c.dot(d))
print(np.dot(c,d))

a = np.random.random((2,4))
np.sum(a)
np.max(a)
np.min(a)
np.mean(a)
np.median(a)

b = np.arange(4).reshape((2,2))
b
np.max(b,axis=1) #默认是列 axis=1表示按行计算
###################################### Lesson4 ######################################
import numpy as np

A=np.arange(2,14).reshape((3,4))
print(A)
np.argmin(A)
np.argmax(A)
np.mean(A)
np.median(A)
np.average(A)
np.cumsum(A) #累加 返回同样的array
np.diff(A)
np.nonzero(A) #返回两个array 第一个是行 第二个是列 表示不是nan和zero的元素的行列
np.sort(A)
A.T
A.T.dot(A)

np.clip(A,3,11) #重要： 小于3的都是3 大于11的都赋值为11
#重要： 几乎所有的numpy的计算 都可以指定axis
#重要： numpy中默认axis=0 表示按照行计算  axis=1 表示按照列计算
###################################### Lesson5 ######################################
import numpy as np

a = np.arange(3,15)
a[3]
a = a.reshape((3,4))
a[2]
a[2,1]
a[2][1] #与a[2,1]相同
a[2,:]
a[:,-1]
a.flatten()

for row in a:
	print(row)

for col in a.T: #迭代列
	print(col)

for item in a.flat: #迭代每个元素
	print(item)
###################################### Lesson6 合并######################################
import numpy as np

a = np.array([1,1,1])
b = np.array([2,2,2])

c=np.vstack((a,b))
print(a.shape,c.shape) #(3,) (2, 3)

d=np.hstack((a,b))
print(a.shape,d.shape) #(3,) (2, 3)

"""
 不能通过a.T 把a转置 可以通过np.newaxis添加一个维度 实现转置
"""
print(a.shape)
print(a.T.shape)
print(a[:,np.newaxis])#这样就可以纵向合并了

e=a[:,np.newaxis]
f=b[:,np.newaxis]
g=np.hstack((e,f))

'''
多个array的合并
'''
x=np.concatenate((e,f,e,f),axis=1)
###################################### Lesson7 分割######################################
import numpy as np

A = np.arange(12).reshape((3,4))
print(np.split(A,2,axis=1))
print(np.split(A,3,axis=0))
print(np.array_split(A,3,axis=1))#可以分割为不同大小的部分  split只能分割为同样的大小

print(np.vsplit(A,3))
print(np.hsplit(A,2))
###################################### Lesson8 赋值######################################
import numpy as np
a=np.arange(4)
b=a
c=a

"""
等于号赋值
view赋值 浅复制
copy赋值 深复制
"""
###################################### Lesson9 pandas######################################
import numpy as np
import pandas as pd

s = pd.Series([1,3,6,np.nan,44,1])
print(s)

dates = pd.date_range('20160101',periods=6)
print(dates)

df = pd.DataFrame(np.random.random((6,4)),index=dates,columns=["a","b","c","d"])
print(df)

df1 = pd.DataFrame(np.arange((12)).reshape((3,4)))
print(df1)

#用字典初始化dataframe
df2 = pd.DataFrame(
	{
		'A':1.,
		'B':pd.Timestamp("20130102"),
		'C':pd.Series(1,index=list(range(4)),dtype='float32'),
		'D':np.array([3]*4,dtype='int32'),
		'E':pd.Categorical(["test","train","test","train"]),
		'F':"foo"
	}
)
df2.dtypes
df2.index
df2.columns
df2.values
df2.describe()
df2.T
df2.sort_index(axis=1,ascending=False)
df2.sort_index(axis=0,ascending=False)
df2.sort_values(by="E")
df2.sort_values(by="E",ascending=False)

###################################### Lesson12 选择数据 切片######################################
import numpy as np
import pandas as pd

dates = pd.date_range("20130101",periods=6)
df = pd.DataFrame(np.arange(24).reshape((6,4)),index=dates,columns=["a","b","c","d"])
print(df["a"],df.a)
print(df[0:3],df["20130101":"20130104"])
print(df.loc["20160101"]) #根据index选择行
print(df.loc[:,["a","b"]]) #选择列
print(df.loc["20130101",["a","b"]])
print(df.iloc[3])#iloc:index locale
print(df.iloc[3:5,1:3])
print(df.iloc[[1,3,5],1:3])
print(df.ix[:3,["a","b"]]) #ix:mixed selection

print(df[df.loc[:,["a"]]>8])#观察和下面有什么不一样 这里是所有的数字》8
print(df[df.a>8])#这里是选择a》8的行
###################################### Lesson13 设置值 ######################################
import numpy as np
import pandas as pd

dates = pd.date_range("20130101",periods=6)
df = pd.DataFrame(np.arange(24).reshape((6,4)),index=dates,columns=["a","b","c","d"])

df.iloc[2,2]=11111
print(df)

df.loc["20130101","b"]=2222
print(df)

df[df.a >4]=0
print(df)
df.a[df.a >4]=0
print(df)
df.b[df.a >4]=0
print(df)

df["F"]=1
print(df)

df["G"]=pd.Series([1,2,3,4,5,6],index=pd.date_range("20130101",periods=6))
print(df)


###################################### Lesson14 处理nan######################################
import numpy as np
import pandas as pd

dates = pd.date_range("20130101",periods=6)
df = pd.DataFrame(np.arange(24).reshape((6,4)),index=dates,columns=["a","b","c","d"])
df.iloc[0,1]=np.nan
df.iloc[1,2]=np.nan

print(df.dropna(axis=0,how="any"))#重要：how={"any","all"}
print(df.dropna(axis=1,how="any"))#重要：how={"any","all"}

print(df.fillna(value=0))
print(df.isnull())
print(np.any(df.isnull())==0)#重要：查询全部数据中是否有nan值
###################################### Lesson15 读取 保存文件数据######################################
import numpy as np
import pandas as pd

data = pd.read_csv("bankChurn.csv")
data.head()
data.to_csv("XXX.csv")
###################################### Lesson16 dataframe 合并 concatenating######################################
import numpy as np
import pandas as pd

########concatenating
df1 = pd.DataFrame(np.ones((3,4))*0,columns=["a","b","c","d"])
df2 = pd.DataFrame(np.ones((3,4))*1,columns=["a","b","c","d"])
df3 = pd.DataFrame(np.ones((3,4))*2,columns=["a","b","c","d"])

res = pd.concat([df1,df2,df3],axis=0)#重要：axis=0竖向的  axis=1是横向的
print(res)
res = pd.concat([df1,df2,df3],axis=0,ignore_index=True)#重要：axis=0竖向的  axis=1是横向的
print(res)

########join,["inner","outer"]
df1 = pd.DataFrame(np.ones((3,4))*0,columns=["a","b","c","d"],index=[1,2,3])
df2 = pd.DataFrame(np.ones((3,4))*1,columns=["b","c","d","e"],index=[2,3,4])
res=pd.concat([df1,df2]) #默认行列没有的话 用NaN填充
print(res)
res2=pd.concat([df1,df2],join="inner",ignore_index=True) #默认行列没有的话 用NaN填充
print(res2)

res3=pd.concat([df1,df2],axis=1,join_axes=[df1.index]) #默认行列没有的话 用NaN填充
print(res3)
res4=pd.concat([df1,df2],axis=1) #默认行列没有的话 用NaN填充
print(res4)

res5 = df1.append(df2,ignore_index=True)
res5 = df1.append([df2,df3],ignore_index=True)
print(res5)

s1 = pd.Series([1,2,3,4],index=["a","b","c","d"])
res6 = df1.append(s1,ignore_index=True)
print(res6)
###################################### Lesson17 merge######################################
import numpy as np
import pandas as pd

left = pd.DataFrame({"key":["K0","K1","K2","K3"],
					 "A": ["A0", "A1", "A2", "A3"],
					 "B": ["B0", "B1", "B2", "B3"]})
right = pd.DataFrame({"key":["K0","K1","K2","K3"],
					 "C": ["C0", "C1", "C2", "C3"],
					 "D": ["D0", "D1", "D2", "D3"]})
res = pd.merge(left=left,right=right,on="key")
print(res)

left = pd.DataFrame({"key1":["K0","K0","K1","K2"],
					"key2":["K0","K1","K0","K1"],
					 "A": ["A0", "A1", "A2", "A3"],
					 "B": ["B0", "B1", "B2", "B3"]})
right = pd.DataFrame({"key1":["K0","K1","K1","K2"],
					  "key2": ["K0", "K0", "K0", "K0"],
					  "C": ["C0", "C1", "C2", "C3"],
					 "D": ["D0", "D1", "D2", "D3"]})
"""
#重要 how:left, right,outer,inner
inner：两个都要存在
outter：在任何一个表中存在即可 如果一个有一个没有 则用NaN填补
left：以left为准
right：以right为准
"""
res = pd.merge(left=left,right=right,on=["key1","key2"],how="inner") #默认合并方法为inner 就是考虑两个都有的情况
print(res)


########indicator
df1 = pd.DataFrame({"col1":[0,1],"col_left":["a","b"]})
df1 = pd.DataFrame({"col1":[1,2,2],"col_right":[2,2,2]})
print(df1)
print(df2)
res = pd.merge(df1,df2,on="col1",how="outer",indicator=True)#indicator 在结果中显示合并时的在那个表中 或者两个都没有
res = pd.merge(df1,df2,on="col1",how="outer",indicator="indicator_column")#indicator 在结果中显示合并时的在那个表中 或者两个都没有

########merge by index
left = pd.DataFrame({"A": ["A0", "A1", "A2"],
					 "B": ["B0", "B1", "B2"]},
					index=["K0","K1","K2"])
right = pd.DataFrame({"C": ["C0", "C2", "C3"],
					 "D": ["D0", "D2", "D3"]},
					 index=["K0","K2","K3"])
res = pd.merge(left,right,on="col1",left_index=True,right_index=True,how="outer")#indicator 在结果中显示合并时的在那个表中 或者两个都没有
res = pd.merge(left,right,on="col1",left_index=True,right_index=True,how="inner")#indicator 在结果中显示合并时的在那个表中 或者两个都没有

########合并时列名冲突
boys =pd.DataFrame({"k":["K0","K1","K2"],"age":[1,2,3]})
girls =pd.DataFrame({"k":["K0","K0","K3"],"age":[4,5,6]})
res = pd.merge(boys,girls,on="k",suffixes=["_boy","_girls"],how="inner")
print(res)

########join和merge类似 用途相似  如果用 自己查
'''
	自己查
	自己查
'''

###################################### Lesson18 pandas 作图 主要看plt######################################
import numpy as np
import pandas as pd

data = pd.Series(np.random.random(1000),index=np.arange(1000))
data = data.cumsum()
data.plot()


# data = pd.DataFrame(np.random.random((1000,4)),index=np.arange(1000),columns=['a','b','c','d'])
data = pd.DataFrame(np.random.random((1000,4)),index=np.arange(1000),columns=list("abcd"))
print(data)
data = data.cumsum()
"""
	plot methods:
	plot bar hist box kde area scatter hexbin pie
"""
data.plot()
data.plot.scatter(x="a",y="b",color="DarkBlue",label="Class 1")




























