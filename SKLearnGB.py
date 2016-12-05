# -*- coding: utf-8 -*-

__author__ = 'Jeonghun Yoon'

import urllib
from sklearn.cross_validation import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
import pylab as plot

# SKLearn library 를 사용하여, Gradient boosting 을 구현하여 보자.
# 먼저 uci archieve 에서 wine 데이터를 가지고 올 것이다.


### 1. Read the wine data from the UCI archive.
xData = []
yData = []
f = urllib.urlopen("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv")

# skip the title (index 1 부터)
lines = f.readlines()[1:]

# If we need title then use this.
'''
lines = f.readlines()
titles = f.pop(0)
'''

for line in lines:
    tokens = line.strip().split(';')
    # extract target values
    yData.append(float(tokens[-1]))
    # extract feature vector
    del (tokens[-1])
    xData.append(map(float, tokens))

### 2. Test set 과 training set 을 나눈다.
# sklearn 의 cross validation 라이브러리를 이용하여 split 한다.
xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size=0.30, random_state=531)

### 3. model training
nTree = 2000
depth = 7
learnRate = 0.01
subSamp = 0.5

GBModel = ensemble.GradientBoostingRegressor(loss='ls', learning_rate=learnRate, n_estimators=nTree, max_depth=depth,
                                             subsample=subSamp)

GBModel.fit(xTrain, yTrain)


### 4. test set 을 통하여 mes 를 구한다.
mseList = []
predictions = GBModel.staged_predict(xTest)
for prediction in predictions:
    mseList.append(mean_squared_error(yTest, prediction))


################# 위의 라인 까지가 train + test 까지이다. test의 결과는 mse 로 도출된다. #################

### 5. 시각화
plot.figure()
# train_score 는 train 단계에서의 MSE 를 말한다.
plot.plot(range(1, nTree + 1), GBModel.train_score_, label = 'Training Set MSE')
plot.plot(range(1, nTree + 1), mseList, label = 'Test Set MSE')
plot.legend(loc='upper right')
plot.xlabel('Number of Trees in Ensemble')
plot.ylabel('Mean Squared Error')
plot.show()