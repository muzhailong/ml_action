import numpy as np
import operator
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def createDataSet():
    group = np.array([1.0, 1, 1], [1.0, 1.0], [0.0], [0.0, 0.1])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classifyo(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis=1)
    distance = sqDistance ** 0.5
    sortedDistIndicies = distance.argsort()
    classCount = {}

    for i in range(k):
        votelabel = labels[sortedDistIndicies[i]]
        classCount[votelabel] = classCount.get(votelabel, 0) + 1
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def file2matrix(filename):
    with open(filename) as fr:
        arrayOLines = fr.readlines()
        numberOfLines = len(arrayOLines)
        returnMat = np.zeros((numberOfLines, 3))
        classLabelVector = []
        index = 0
        for line in arrayOLines:
            line = line.strip()
            listPromLine = line.split('\t')
            returnMat[index, :] = listPromLine[0:3]
            classLabelVector.append(listPromLine[-1])
            index += 1
    return returnMat, LabelEncoder().fit_transform(classLabelVector)


def autoNorm(dataSet):
    minValue = dataSet.min(axis=0)
    maxValue = dataSet.max(axis=0)
    ranges = maxValue - minValue
    rowNum = dataSet.shape[0]
    normDataSet = (dataSet - np.tile(minValue, (rowNum, 1))) / np.tile(ranges, (rowNum, 1))
    return normDataSet, ranges, minValue


def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix("datingTestSet.txt")
    normMat, ranges, minValues = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0

    for i in range(numTestVecs):
        classifierResult = classifyo(normMat[i, :], normMat[numTestVecs:m, :], \
                                     datingLabels[numTestVecs:],3)
        if classifierResult != datingLabels[i]:
            errorCount += 1

    print(errorCount / m)

datingClassTest()