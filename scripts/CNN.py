import tensorflow as tf

class CNN(object):

    def __init__(self, outputsNumber, convLayersNumber, inputImgSize, filtersShape, bias):
        if convLayersNumber != len(filtersShape):
            raise Exception
        self.OUTPUTS = outputsNumber
        self.CONV_LAYERS = convLayersNumber
        self.INPUT_SHAPE = inputImgSize
        self.BIAS = bias
        self.FILTERS_SHAPE = filtersShape
        self.xPlaceholder = tf.placeholder("float", [None] + list(inputImgSize) + [3])
        self.yPlaceholder = tf.placeholder("float", [None, outputsNumber])
        self.keep_prob = tf.placeholder(tf.float32)
        self.WEIGHTS = self.initializeWeights()

    def setOutputsNumber(self, number):
        self.OUTPUTS = number

    def initializeWeights(self):
        weights = {
            'wd1': tf.get_variable('W3', shape=(4*4*64, 64), initializer=tf.contrib.layers.xavier_initializer()),
            'out': tf.get_variable('W6', shape=(64, self.OUTPUTS), initializer=tf.contrib.layers.xavier_initializer()),
        }
        for i in range(0, self.CONV_LAYERS):
            weights['wc' + str(i + 1)] = tf.get_variable('W' + str(i), shape=self.FILTERS_SHAPE[i], initializer=tf.contrib.layers.xavier_initializer())
        return weights

    def conv2d(self, inputs_batch, filters, b, strides=1):
        # conv2d, bias and relu activation
        inputs_batch = tf.nn.conv2d(inputs_batch, filters, strides=[1, strides, strides, 1], padding='SAME')
        inputs_batch = tf.nn.bias_add(inputs_batch, b)
        return tf.nn.relu(inputs_batch)

    def maxpool2d(self, x, k=2):
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    def conv_net(self, input, keep_prob):
        # first convolutional layer
        conv1 = self.conv2d(input, self.WEIGHTS['wc1'], self.BIAS['bc1'])
        # Pooling with filter size 2x2, outputs a 8x8 (down sampled 16x16).
        conv1 = self.maxpool2d(conv1, k=2)

        # second convolutional layer
        conv2 = self.conv2d(conv1, self.WEIGHTS['wc2'], self.BIAS['bc2'])
        # Pooling with filter size 2x2, outputs a 4x4 (down sampled 8x8).
        conv2 = self.maxpool2d(conv2, k=2)

        # Fully connected layer
        fc1 = tf.reshape(conv2, [-1, self.WEIGHTS['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, self.WEIGHTS['wd1']), self.BIAS['bd1'])
        fc1 = tf.nn.relu(fc1)
        # Dropout for some neurons
        dropout = tf.nn.dropout(fc1, keep_prob)
        # Output, class prediction
        out = tf.add(tf.matmul(dropout, self.WEIGHTS['out']), self.BIAS['out'])
        return out

    def run(self, session, fetches, batchX, batchY, keep_prob):
        return session.run(fetches, feed_dict={self.xPlaceholder: batchX, self.yPlaceholder: batchY, self.keep_prob: keep_prob})