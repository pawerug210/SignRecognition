import numpy as np


def createClassifiedAsList(expectedAndClassified):
    classifiedAsDict = {}
    for _class in np.unique(expectedAndClassified[:, 0]):
        classMask = [x == _class for x in expectedAndClassified[:, 0]]
        classifiedAs = [value for idx, value in enumerate(expectedAndClassified[:, 1]) if classMask[idx]]
        classifiedAsDict[_class] = classifiedAs
    return classifiedAsDict
