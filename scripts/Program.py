import Common
from DataPreprocessing import resize
from DataPreprocessing import labelsToOutputs
from DataPreprocessing import createMapping
import tensorflow as tf
import CNN

#training settings
iterations = 1000
batchSize = 32
learningRate = 0.001

#testing settings
testSamplesNumber = 20

#CNN settings
classes = range(0, 2)
samplesNumber = 20
inputImgSize = (16, 16)
# filter (3x3), 3 channels (RGB), number of filters applied
filtersShape = [(3, 3, 3, 32),
                (3, 3, 32, 64),
                (3, 3, 64, 128)]

biases = {
    'bc1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'bc2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'bc3': tf.get_variable('B2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'bd1': tf.get_variable('B3', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('B4', shape=(len(classes)), initializer=tf.contrib.layers.xavier_initializer()),
}

kwargs = {'inputImgSize': inputImgSize,
          'outputsNumber': len(classes),
          'convLayersNumber': 3,
          'filtersShape': filtersShape,
          'bias': biases,
          'learningRate': learningRate
          }

# test
images, labels = Common.readTrafficSigns('../Data/GTSRB_Final_Training_Images/GTSRB/Final_Training/Images', classes, samplesNumber)
images = resize(images, inputImgSize)
outputs = labelsToOutputs(classes, labels)

# test analysis
# Common.displayImagesSample(images)
# Common.displayLabelsHist(labels)

# train
testImages, testLabels = Common.readTestImages('../Data/GTSRB_Final_Test_Images/GTSRB/Final_Test/Images', classes, testSamplesNumber)
testImages = resize(testImages, inputImgSize)
testOutputs = labelsToOutputs(classes, testLabels)

if len(testImages) != len(testOutputs):
    raise Exception

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
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(cNN.yPlaceholder, 1))
# calculate accuracy across all the given images and average them out.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    summary_writer = tf.summary.FileWriter('./Output', session.graph)
    for i in range(iterations):
        for k in range(len(batchesX)):
            trainBatchX = batchesX[k]
            trainBatchY = batchesY[k]
            op = cNN.run(session, optimizer, trainBatchX, trainBatchY)
            loss, acc = cNN.run(session, [cost, accuracy], trainBatchX, trainBatchY)
        print("Iter " + str(i) + ", Loss= " + \
              "{:.6f}".format(loss) + ", Training Accuracy= " + \
              "{:.5f}".format(acc))
        test_acc, valid_loss = cNN.run(session, [accuracy, cost], testImages, testOutputs)
        print("Testing Accuracy:", "{:.5f}".format(test_acc))


