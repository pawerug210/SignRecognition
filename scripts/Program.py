import Common
from DataPreprocessing import resize
from DataPreprocessing import labelsToOutputs
from DataPreprocessing import createMapping
import Helper as help
import tensorflow as tf
import CNN

#training settings
epochs = 50
batchSize = 512
learningRate = 0.001

#testing settings
testSamplesNumber = 50

#CNN settings
classes = range(10, 30)
samplesNumber = 200
inputImgSize = (16, 16)
# filter (3x3), 3 channels (RGB), number of filters applied
filtersShape = [(3, 3, 3, 32),
                (3, 3, 32, 64)]

biases = {
    'bc1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'bc2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'bd1': tf.get_variable('B2', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('B3', shape=(len(classes)), initializer=tf.contrib.layers.xavier_initializer()),
}

kwargs = {'inputImgSize': inputImgSize,
          'outputsNumber': len(classes),
          'convLayersNumber': 2,
          'filtersShape': filtersShape,
          'bias': biases,
          'learningRate': learningRate
          }

# test
images, labels = Common.readTrafficSigns('../Data/GTSRB_Final_Training_Images/GTSRB/Final_Training/Images', classes, samplesNumber)
images = resize(images, inputImgSize)
outputs = labelsToOutputs(classes, labels)

images, outputs = Common.shuffle(images, outputs)
# analysis
# Common.displayImagesSample(images)
# Common.displayLabelsHist(labels)

# train
testImages, testLabels = Common.readTestImages('../Data/GTSRB_Final_Test_Images/GTSRB/Final_Test/Images', classes, testSamplesNumber)
testImages = resize(testImages, inputImgSize)
testOutputs = labelsToOutputs(classes, testLabels)

# Common.displayImagesSize(images, testImages)
# Common.displayLabelsHist(testLabels)
# Common.displayImagesSample(testImages)

if len(testImages) != len(testOutputs):
    raise Exception
if not all([any(x) for x in testOutputs]):
    raise ValueError('Not all outputs have value')


classesMapping = createMapping(classes)

cNN = CNN.CNN(**kwargs)
batchesX = [images[batch: batch + batchSize] for batch in range(0, len(images) - batchSize, batchSize)]
batchesY = [outputs[batch: batch + batchSize] for batch in range(0, len(outputs) - batchSize, batchSize)]

if len(batchesX) != len(batchesY):
    raise Exception

pred = cNN.conv_net(cNN.xPlaceholder)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=cNN.yPlaceholder))
optimizer = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(cost)
# Here you check whether the index of the maximum value of the predicted image is equal to the actual labelled image. and both will be a column vector.
# todo specify variable so it is evaluated only once /?/
classified_indexes = tf.argmax(pred, 1)
expected_indexes = tf.argmax(cNN.yPlaceholder, 1)
expectedAndClassified = tf.stack([expected_indexes, classified_indexes], axis=1)
correct_prediction = tf.equal(expected_indexes, classified_indexes)
# calculate accuracy across all the given images and average them out.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    train_loss_global = []
    train_accuracy_global = []
    test_loss = []
    test_accuracy = []
    summary_writer = tf.summary.FileWriter('./Output', session.graph)
    for i in range(epochs):
        train_loss = []
        train_accuracy = []
        for k in range(len(batchesX)):
            trainBatchX = batchesX[k]
            trainBatchY = batchesY[k]
            op = cNN.run(session, optimizer, trainBatchX, trainBatchY)
            loss, acc = cNN.run(session, [cost, accuracy], trainBatchX, trainBatchY)
            # print(help.createClassifiedAsList(expToClass))
            train_accuracy.append(acc)
            train_loss.append(loss)
        avg_train_loss = sum(train_loss) / len(train_loss)
        avg_train_acc = sum(train_accuracy) / len(train_accuracy)
        print("Epoch " + str(i) + ", Loss= " + \
              "{:.6f}".format(avg_train_loss) + ", Training Accuracy= " + \
              "{:.5f}".format(avg_train_acc))
        valid_loss, test_acc, expToClass = cNN.run(session, [cost, accuracy, expectedAndClassified], testImages, testOutputs)
        print("Testing Accuracy:", "{:.5f}".format(test_acc))
        print("Testing Loss:", "{:.5f}".format(valid_loss))
        train_accuracy_global.append(avg_train_acc * 100.0)
        train_loss_global.append(avg_train_loss)
        test_accuracy.append(test_acc * 100.0)
        test_loss.append(valid_loss)
Common.displayClassificationPlot(help.createClassifiedAsList(expToClass))
Common.displayResults(epochs, train_accuracy_global, train_loss_global, test_accuracy, test_loss)



