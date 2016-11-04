#-*- coding: utf-8 -*-
# Gardient Boosting

'''
Gradient Boosting vs Bagging(Bootstrap Aggregation) vs Random Forest를 비교해보자.
여기서는 synthetic data를 사용하여, GB를 먼저 공부해 볼 것이다.
이 코드는 학습을 위한 것이다.
'''

__author__ = 'Jeonghun Yoon'

import numpy as np
import matplotlib.pyplot as plot
from sklearn import tree

### 1. synthetic data를 생성할 것이다. y_i = x_i + alpha가 되도록 data를 1,000개 생성할 것이다.
nPoints = 1000

# x의 값은, -0.5와 0.5사이로 정한다.
xData = [(float(i)/float(nPoints) - 0.5) for i in xrange(nPoints)]

# y의 값은, y_i = x_i + alpha가 될 것이다. alpha는 normal gaussian dist를 따르며, 표준편차는 0.1로 한다.
# 이 경우, 분산은 최대 0.01이 될 것이다.
np.random.seed(1)
yData = [x+np.random.normal(scale = 0.1) for x in xData]

# data set의 30%를 test set으로 정한다.
nSamples = int(0.3 * len(xData))
testIdx = np.random.choice(range(nPoints), nSamples, replace = False)
testIdx.sort()
trainIdx = [i for i in range(nPoints) if i not in testIdx]

xTest = [xData[i] for i in testIdx]
yTest = [yData[i] for i in testIdx]
xTrain = [xData[i] for i in trainIdx]
yTrain = [yData[i] for i in trainIdx]

# Sk-learn을 사용하기 위하여 input 형태를 바꾼다.
xTest = np.array(xTest).reshape(-1, 1)
yTest = np.array(yTest).reshape(-1, 1)
xTrain = np.array(xTrain).reshape(-1, 1)
yTrain = np.array(yTrain).reshape(-1, 1)

### 2. Ensemble 모델을 생성하기 위한 셋팅을 한다.
# 모델 셋팅 시, 입력해야할 파라미터 1
nTreeMax = 30
# 모델 셋팅 시, 입력해야할 파라미터 2
treeDepth = 5
# 모델 셋팅 시, 입력해야할 파라미터 3
eps = 0.3

modelList = []
predList = []

# Bagging, Random forest와 다르게 target이 residual이다.
residuals = yTrain

for iTree in range(nTreeMax):
    modelList.append(tree.DecisionTreeRegressor(max_depth=treeDepth))
    # RegressionTree를 fit(learn) 시킨다. target value가 y가 아니고 residuals임에 주의하자.
    modelList[-1].fit(xTrain, residuals)

    # 현재의 tree를 이용하여, 잔차를 구하기 위하여, prediction을 한다
    prediction = modelList[-1].predict(xTrain)
    # residuals를 업데이트 한다.
    residuals = [residuals[i] - eps *  prediction[i] for i in range(len(residuals))]

    # ensemble을 위하여 현재 tree의 prediction을 저장한다.
    latestPrediction = modelList[-1].predict(xTest)
    predList.append(list(latestPrediction))

# 최초 n개의 모델에서 누적한 예측 생성
mse = []
allPrediction = []

for iModel in range(len(modelList)):
    prediction = []

    # 예측을 한다. 각 test data에 대한 예측값을 더해준다.
    for iPred in range(len(xTest)):
        prediction.append(sum([predList[i][iPred] for i in range(iModel + 1)]) * eps)
    allPrediction.append(prediction)

    # mse를 구한다.
    errors = yTest - np.array(prediction).reshape(-1, 1)
    mse.append(sum(errors * errors) / len(yTest))

nModels = [i+1 for i in range(len(modelList))]

plot.plot(nModels, mse)
plot.axis('tight')
plot.xlabel('Number of Models in Ensemble')
plot.ylabel('Mean Squared Error')
plot.ylim((0.0, max(mse)))
plot.show()

plotList = [0, 14, 29]
lineType = [':', '-.', '--']
plot.figure()
for i in range(len(plotList)):
    iPlot = plotList[i]
    textLegend = 'Prediction with ' + str(iPlot) + ' Trees'
    plot.plot(xTest, allPrediction[iPlot], label = textLegend, linestyle = lineType[i])
plot.plot(xTest, yTest, label = 'True y Value', alpha=0.25)
plot.legend(bbox_to_anchor=(1, 0.3))
plot.axis('tight')
plot.xlabel('x value')
plot.ylabel('Prediction')
plot.show()

