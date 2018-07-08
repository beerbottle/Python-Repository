from __future__ import division
from numpy.random import randn
import numpy as np
import os
import matplotlib.pyplot as plt
np.random.seed(12345)
plt.rc('figure', figsize=(10, 6))
from pandas import Series, DataFrame
import pandas as pd
np.set_printoptions(precision=4)

df = DataFrame({'key1' : ['a', 'a', 'b', 'b', 'a'],
                'key2' : ['one', 'two', 'one', 'two', 'one'],
                'data1' : np.random.randn(5),
                'data2' : np.random.randn(5)})
df
grouped = df['data1'].groupby(df['key1'])
grouped
grouped.mean()

for name, group in df.groupby('key1'):
    print(name)
    print(group)

pieces = dict(list(df.groupby('key1')))
pieces['b']

type(list(df.groupby('key1')))

s_grouped = df.groupby(['key1', 'key2'])['data2']
s_grouped


people = DataFrame(np.random.randn(5, 5),
                   columns=['a', 'b', 'c', 'd', 'e'],
                   index=['Joe', 'Steve', 'Wes', 'Jim', 'Travis'])
people.ix[2:3, ['b', 'c']] = np.nan
people
mapping = {'a': 'red', 'b': 'red', 'c': 'blue',
           'd': 'blue', 'e': 'red', 'f' : 'orange'}
people.groupby(len).sum()

grouped = df.groupby('key1')
grouped[['data1', 'data2']].quantile(0.9)

# test






