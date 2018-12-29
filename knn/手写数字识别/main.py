import numpy as np
import os
import knn.knn as knn


def img2vector(filename):
    with open(filename) as fp:
        s = ""
        for line in fp.readlines():
            s += line.strip()
        tempList = list(map(int, list(s)))
        return np.array(tempList)


def handwritingClassTest():
    hwLabels = []
    trainingFileList = os.listdir("trainingDigits")
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))

    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split(".")[0]
        classNumStr = int(fileStr.split("_")[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector("trainingDigits/%s" % fileNameStr)
    testFileList = os.listdir("testDigits")
    errorCount = 0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split(".")[0]
        classNumStr = int(fileStr.split("_")[0])
        vectorUnderTest = img2vector("testDigits/%s" % fileNameStr)
        classifierResult = knn.classifyo(vectorUnderTest, trainingMat, hwLabels, 3)
        if classifierResult != classNumStr:
            errorCount += 1
    print(errorCount / mTest)


if __name__ == "__main__":
    handwritingClassTest()
