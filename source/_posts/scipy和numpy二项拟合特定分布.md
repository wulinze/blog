---
title: scipy和numpy二项拟合特定分布
date: 2024-01-10 08:40:37
tags:
---
## 通过泰勒展开的方式拟合特定分布
* 起因
线上采集了一组数据，但是数据分布情况不是标准正态分布，而且是一个双峰数据，考虑通过最小二乘法的方式对于分布进行拟合。主要使用工具numpy和scipy
``` python
from scipy.stats import rv_continuous
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import numpy.polynomial.polynomial as poly
import os
import math
import glob
from scipy import integrate
from scipy.integrate import quad
from scipy import optimize
import copy

class MyDistribution(rv_continuous):
    def __init__(self, data, bins_num=None):
        super(MyDistribution, self).__init__()
        data_min = np.min(data)
        data_max = np.max(data)
        self._data_min = data_min
        self._data_max = data_max
        if bins_num is None:
            bins_num = max(len(data) // 10, (data_max - data_min) // 10)

        p, bins = np.histogram(data, bins=bins_num, range=(data_min, data_max))
        p = self.norm(p)

        self._z1 = poly.polyfit(bins[: -1], p, 30)
        self._p1 = poly.Polynomial(self._z1)

        # f = lambda x: (self._p1(x) if self._p1(x) > 0 else 0)
        f = lambda x: (0 if (x < data_min or x > data_max) else (self._p1(x) if self._p1(x) > 0 else 0))

        k, _ = integrate.quad(f, data_min, data_max)
        self._f = lambda x: (f(x)/k)

        ax = plt.subplot()
        xlist = np.linspace(data_min-1, data_max+1, num=bins_num, endpoint=True)
        pdflist = list()
        for x in xlist:
            pdflist.append(self.pdf(x))
            # pdflist.append(f(x))

        ax.plot(xlist, pdflist, color='royalblue')
        ax.hist(bins[: -1], bins=len(bins[: -1]), weights=p)
        

    def cdf_inverse(self, x):
        def function(y):
            return self._cdf(y) - x

        result = optimize.root(function, x0=self._data_min)
        inverse = result.x[0]
        return inverse

    def norm(self, x):
        ans = []
        x_sum = np.sum(x)
        for i in range(len(x)):
            ans.append(x[i]/x_sum)
        
        return ans

    def _rvs(self, *args, **kwargs):
        size = kwargs.get('size', 1)
        if len(size) == 0:
            size = 1
        else:
            size = size[0]
        random_samples = np.empty(size)
        for i in range(size): 
            u = np.random.uniform()
            random_samples[i] = self.cdf_inverse(u)
        return random_samples

    def _pdf(self, x):
        return self._f(x)

    def _cdf(self, x):
        if x < self._data_min:
            return 0
        elif x > self._data_max:
            return 1
        cdf, _ = quad(self._pdf, self._data_min, x)

        return cdf
```
简单说一下代码，scipy内部提供了对于连续分布的基本信息的类rv_continuous，还有用于离散变量的rv_discrete。具体使用方式可以看官方文档，英文看不懂就去github找中文的。里面提供了一些基本的以_开头的函数，通过重载函数可以实现自定义分布，pdf为概率密度函数，cdf为累积分布函数。_rvs负责采样的逻辑。需要自己实现一个求逆函数(反函数)的操作。
* 函数求逆函数(反函数)
函数求反函数就是初始认为函数f(x)=y找到一个g(y)=x的函数g，通过scipy中optimize.root可以直接求出。
``` python
def cdf_inverse(self, x):
        def function(y):
            return self._cdf(y) - x

        result = optimize.root(function, x0=self._data_min)
        inverse = result.x[0]
        return inverse
```