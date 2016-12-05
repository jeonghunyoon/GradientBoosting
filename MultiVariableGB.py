# -*- coding: utf-8 -*-
# gradient boostring with multi variable

__author__ = 'Jeonghun Yoon'

import urllib
import random
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plot

'''
I load red wine quality data from the uci repository and
implement the gradient boosting with multi variables
'''

### 1. Read the wine data from the UCI archive.
xData = []
yData = []
f = urllib.urlopen("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv")

# skip the title
lines = f.readlines()[1:]

# If we need title then use this.
'''
lines = f.readlines()
titles = f.pop(0)
for line in lines:
'''

for line in lines:
    tokens = line.strip().split(';')
    # extract target values
    yData.append(float(tokens[-1]))
    # extract feature vector
    del (tokens[-1])
    xData.append(map(float, tokens))

### 2. Divide data set into two parts, train set(70%), test set(30%)
nData = len(xData)
nTest = int(nData * 0.3)
random.seed(1)

# test set은 랜덤하게 선택한다.
testIdx = random.sample(range(nData), nTest)
testIdx.sort()
trainIdx = [idx for idx in range(nData) if idx not in testIdx]

xTest = [xData[idx] for idx in testIdx]
yTest = [yData[idx] for idx in testIdx]
xTrain = [xData[idx] for idx in trainIdx]
yTrain = [yData[idx] for idx in trainIdx]

nTrain = len(xTrain)

### 3. Set parameters for regression tree and learn the model.
maxDepth = 5
maxTree = 30
eps = 0.3

# I will put the 30 base learner this list.
modelList = []
# At each step, I will put the predict result on this.
predList = []
# In Gradient Boosting, models are fitted using residuals between predict values and target values.
residuals = yTrain

for iModel in range(maxTree):
    model = tree.DecisionTreeRegressor(max_depth=maxDepth)
    # residuals를 이용하여 fitting
    model.fit(xTrain, residuals)

    # xTrain의 predict 값을 구한다.
    predTrain = [model.predict(np.array(xTrain[i]).reshape(1, -1)) for i in range(nTrain)]
    # residuals를 업데이트 하기 위하여 xTrain의 predict 값을 구하고, eps의 비율만큼을 현재 residual에서 줄인다.
    # eps 는 gradient 의 learning rate 이다.
    residuals = [residuals[i] - eps * predTrain[i] for i in range(nTrain)]

    # xTest의 predict 값을 구한다. 이것은 나중에 test set의 target values와의 mse를 구하기 위해서이다.
    predTest = [model.predict(np.array(xTest[i]).reshape(1, -1)) for i in range(nTest)]

    # model 및 predTest를 각각 넣는다.
    modelList.append(model)
    predList.append(predTest)

mse = []
allPrediction = []

for iModel in range(maxTree):
    prediction = []
    # prediction의 값을 테스트 해보기 위하여, 마지막 n번째 모델까지의 예측합을 구한다.
    for iPred in range(nTest):
        prediction.append(sum([predList[i][iPred] for i in range(iModel + 1)]) * eps)
    allPrediction.append(prediction)

    error = np.array(yTest).reshape(-1, 1) - np.array(prediction)
    mse.append(sum(error * error) / nTest)


nModels = [i+1 for i in range(len(modelList))]

# mse의 plot을 찍어본다.
plot.plot(nModels, mse)
plot.axis('tight')
plot.xlabel('Number of Models in Ensemble')
plot.ylabel('Mean Squared Error')
plot.ylim((0.0, max(mse)))
plot.show()

