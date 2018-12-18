import tensorflow as tf


# todo builder
class CNN(object):

    def __init__(self, outputsNumber, convLayersNumber, inputImgSize, filtersShape, bias, learningRate):
        if convLayersNumber != len(filtersShape):
            raise Exception
        self.OUTPUTS = outputsNumber
        self.CONV_LAYERS = convLayersNumber
        self.INPUT_SHAPE = inputImgSize
        self.BIAS = bias
        self.FILTERS_SHAPE = filtersShape
        self.xPlaceholder = tf.placeholder("float", [None] + list(inputImgSize) + [3])
        self.yPlaceholder = tf.placeholder("float", [None, outputsNumber])

        self.WEIGHTS = self.initializeWeights()
        pass

    def setOutputsNumber(self, number):
        self.OUTPUTS = number

    def initializeWeights(self):
        weights = {
            'wd1': tf.get_variable('W3', shape=(2*2*128, 128), initializer=tf.contrib.layers.xavier_initializer()),
            'out': tf.get_variable('W6', shape=(128, self.OUTPUTS), initializer=tf.contrib.layers.xavier_initializer()),
        }
        for i in range(0, self.CONV_LAYERS):
            weights['wc' + str(i + 1)] = tf.get_variable('W' + str(i), shape=self.FILTERS_SHAPE[i], initializer=tf.contrib.layers.xavier_initializer())
        return weights

    def conv2d(self, inputs_batch, filters, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        inputs_batch = tf.nn.conv2d(inputs_batch, filters, strides=[1, strides, strides, 1], padding='SAME')
        inputs_batch = tf.nn.bias_add(inputs_batch, b)
        return tf.nn.relu(inputs_batch)

    def maxpool2d(self, x, k=2):
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    def conv_net(self, input):
        filters = self.WEIGHTS
        biases = self.BIAS
        # here we call the conv2d function we had defined above and pass the input image x, weights wc1 and bias bc1.
        conv1 = self.conv2d(input, filters['wc1'], biases['bc1'])
        # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 14*14 matrix.
        conv1 = self.maxpool2d(conv1, k=2)

        # Convolution Layer
        # here we call the conv2d function we had defined above and pass the input image x, weights wc2 and bias bc2.
        conv2 = self.conv2d(conv1, filters['wc2'], biases['bc2'])
        # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 7*7 matrix.
        conv2 = self.maxpool2d(conv2, k=2)

        conv3 = self.conv2d(conv2, filters['wc3'], biases['bc3'])
        # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 4*4.
        conv3 = self.maxpool2d(conv3, k=2)

        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(conv3, [-1, filters['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, filters['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        # Output, class prediction
        # finally we multiply the fully connected layer with the weights and add a bias term.
        out = tf.add(tf.matmul(fc1, filters['out']), biases['out'])
        return out

    def run(self, session, fetches, batchX, batchY):
        return session.run(fetches, feed_dict={self.xPlaceholder: batchX, self.yPlaceholder: batchY})