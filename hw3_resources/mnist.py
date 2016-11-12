import numpy as np

from matplotlib import pyplot as plt

from NN import *

def buildTrainingSet():
    return buildSet(200, 0)
def buildValidationSet():
    return buildSet(150, 200)
def buildTestSet():
    return buildSet(150, 350)

def buildSet(num, used):
    X = np.array([]).reshape(0,784)
    Y = np.array([]).reshape(0,1)
    for digit in xrange(10):
        X = np.concatenate((X,loadDigit(digit, num, used)))
        Y = np.concatenate((Y,digit * np.ones((num, 1))))
    return X,Y
    
def loadDigit(digit, num, used):
    data = np.loadtxt('data/mnist_digit_' + str(digit) + '.csv')
    X = data[used:used+num]
    for x in X:
        for i in xrange(len(x)):
            x[i] = 2.0*x[i] / 255.0 - 1
            # x[i] = x[i]
    return X

def evaluatePredict(X, Y, predict):
    n = len(X)
    misclass = 0
    misclassifications = []
    for i in xrange(n):
        x = X[i]
        y = Y[i]
        y_pred = predict(x)
        if y != y_pred:
            misclass += 1
            misclassifications.append((y, y_pred))

    misclassrate = misclass*1.0/n
    # print "# of misclass: ", misclass
    # print "Accuracy rate: ", 1-misclassrate
    # print "Misclassifications: ", misclassifications
    return misclassrate, misclassifications

def buildNNPredict(X,Y,Xv,Yv, l):
    n,d = X.shape
    layers = [d, 20,20, 10]
    nn = Neural_Network(layers)
    nn.train(X,Y,Xv,Yv,l, max_iter=500000)
    predict = nn.predict
    return predict

def openMisclassifications(X,Y,misclassifications):
    for i in misclassifications:
        data = X[i].reshape((28,28))
        plt.imshow(data, cmap = plt.get_cmap('gray'))
        plt.show()

ls = [0.0001,0.001,0.01, 0.1, 1, 10]
# ls = [0.01]

print '======Setting Up======'
trainX,trainY = buildTrainingSet()
valX, valY = buildValidationSet()
testX, testY = buildTestSet()

print '======Training======'
predicts = []
for l in ls:
    predict = buildNNPredict(trainX, trainY, valX, valY, l)
    predicts.append((l, predict))
    
    misclassifications = evaluatePredict(trainX, trainY, predict)

# openMisclassifications(trainX, trainY, misclassifications)

print '======Validation======'
bestParam = 0
bestpredict = 0
bestmisclassifications = []
bestmisclassrate = 1
for param,predict in predicts:
    misclassrate, misclassifications = evaluatePredict(valX, valY, predict)
    if misclassrate < bestmisclassrate:
        bestParam = param
        bestpredict = predict
        bestmisclassifications = misclassifications
        bestmisclassrate = misclassrate

print '======Best Validation======'
print "Param: ", bestParam
print "Misclassification rate: ", bestmisclassrate
# print "Misclassifications: ", bestmisclassifications
# openMisclassifications(valX, valY, bestmisclassifications)

print '======Testing======'
misclassrate,misclassifications = evaluatePredict(testX, testY, bestpredict)
print "Misclassification rate: ", misclassrate
# openMisclassifications(testX, testY, misclassifications)

