import matplotlib.pyplot as plt
from DataPreprocessing import crop
from collections import Counter
import numpy as np
import math
import csv
import random


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
    columns = math.ceil(math.sqrt(len(images)))
    rows = columns
    iterations = columns * rows + 1
    images = images[:iterations]
    for i in range(1, iterations):
        if i > len(images):
            break
        img = images[i - 1]
        fig.add_subplot(rows, columns, i)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img)
    plt.show()


def displayLabelsHist(labels):
    labelsCountsDict = Counter(labels)
    x = list(map(int, labelsCountsDict.keys()))
    y = list(labelsCountsDict.values())
    plt.xlabel('Class id')
    plt.ylabel('Samples number')
    plt.bar(x, y)
    plt.xticks(range(0, len(x)))
    plt.show()


def displayImagesSize(trainingImages, testImages):
    plt.plot([x.shape[0] for x in trainingImages], [x.shape[1] for x in trainingImages], 'r.', label='train')
    plt.plot([x.shape[0] for x in testImages], [x.shape[1] for x in testImages], 'b.', label='test')
    plt.xticks(range(0, 220, 5))
    plt.xlabel('width [px]')
    plt.ylabel('height [px]')
    plt.legend(loc='upper left')
    plt.show()


def displayResults(epochs, train_accuracy, train_loss, test_accuracy, test_loss):
    plt.subplot(1, 2, 1)
    plt.plot(range(0, epochs), train_loss, 'b', label='train')
    plt.plot(range(0, epochs), test_loss, 'r', label='test')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss')
    plt.legend(loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(range(0, epochs), train_accuracy, 'b', label='train')
    plt.plot(range(0, epochs), test_accuracy, 'r', label='test')
    plt.xlabel('epoch')
    plt.ylabel('accuracy [%]')
    plt.title('Accuracy')
    plt.legend(loc='upper left')

    plt.show()

def displayClassificationPlot(expectedToClassifiedMap):
    correct = []
    wrong = []
    classes = []
    for _class in expectedToClassifiedMap.keys():
        classCorrect = expectedToClassifiedMap[_class].count(_class) / len(expectedToClassifiedMap[_class])
        classWrong = 1.0 - classCorrect
        correct.append(classCorrect)
        wrong.append(classWrong)
        classes.append(_class)
    barWidth = 0.2
    indexes = np.arange(len(correct))
    plt.bar(indexes, correct, barWidth, color='g')
    plt.bar(indexes + barWidth, wrong, barWidth, color='r')
    plt.xticks(indexes, classes)
    plt.show()

def shuffle(images, labels):
    combined = list(zip(images, labels))
    random.shuffle(combined)

    images[:], labels[:] = zip(*combined)
    return images, labels