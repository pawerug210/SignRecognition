import matplotlib.pyplot as plt
from DataPreprocessing import crop
from collections import Counter
import csv


def readTrafficSigns(rootpath, classes, samplesNumber=None):
    images = []
    labels = []
    for c in classes:
        prefix = rootpath + '/' + format(c, '05d') + '/'
        annotationsFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv')
        reader = csv.reader(annotationsFile, delimiter=';')
        next(reader, None)  # skip header
        counter = 0
        for row in reader:
            counter += 1
            images.append(crop(image=plt.imread(prefix + row[0]),
                               startX=int(row[3]),
                               startY=int(row[4]),
                               stopX=int(row[5]),
                               stopY=int(row[6])))
            labels.append(row[7])
            if samplesNumber is not None and counter == samplesNumber:
                break
        annotationsFile.close()
    return images, labels


def readTestImages(rootpath, classes, samplesNumber=None):
    # todo extract logic for reading pictures
    images = []
    labels = []
    counterMap = {str(val): 0 for val in classes}
    annotationsFile = open(rootpath + '/' + 'GT-final_test.csv')
    reader = csv.reader(annotationsFile, delimiter=';')
    next(reader, None)  # skip header
    for row in reader:
        pictureClass = row[7]
        if int(pictureClass) in list(classes) and counterMap[pictureClass] < samplesNumber:
            counterMap[pictureClass] += 1
            images.append(crop(image=plt.imread(rootpath + '/' + row[0]),
                               startX=int(row[3]),
                               startY=int(row[4]),
                               stopX=int(row[5]),
                               stopY=int(row[6])))
            labels.append(pictureClass)
    annotationsFile.close()
    return images, labels



def displayImagesSample(images):
    fig = plt.figure(figsize=(8, 8))
    columns = 5
    rows = 5
    iterations = columns * rows + 1
    images = images[:iterations]
    for i in range(1, iterations):
        img = images[i - 1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()


def displayLabelsHist(labels):
    labelsCountsDict = Counter(labels)
    x = list(map(int, labelsCountsDict.keys()))
    y = list(labelsCountsDict.values())
    plt.bar(x, y)
    plt.xticks(range(0, len(x)))
    plt.show()