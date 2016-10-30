#-*- coding: utf-8 -*-
# Gardient Boosting

'''
Gradient Boosting vs Bagging(Bootstrap Aggregation) vs Random Forest를 비교해보자.
여기서는 synthetic data를 사용하여, GB를 먼저 공부해 볼 것이다.
이 코드는 학습을 위한 것이다.
'''

import numpy as np

__author__ = 'Jeonghun Yoon'



### 1. synthetic data를 생성할 것이다. y_i = x_i + alpha가 되도록 data를 1,000개 생성할 것이다.
nPoints = 1000

# x의 값은, -0.5와 0.5사이로 정한다.
xData = [(float(i)/float(nPoints) - 0.5) for i in xrange(nPoints)]

# y의 값은, y_i = x_i + alpha가 될 것이다. alpha는 normal gaussian dist를 따르며, 표준편차는 0.1로 한다.
# 이 경우, 분산은 최대 0.01이 될 것이다.
np.random.seed(1)
yData = [x+np.random.normal(scale = 0.1) for x in xData]

