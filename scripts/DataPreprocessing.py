import scipy.misc
import tensorflow as tf


def crop(image, startX, stopX, startY, stopY):
    return image[startX: stopX + 1, startY: stopY + 1, :]  # slice end border inclusive ( + 1 )


def resizeTF(images, size):
    images = tf.convert_to_tensor(images)
    return tf.image.resize_images(images, size)


def resize(images, size):
    resizedImages = []
    for image in images:
        resizedImages.append(scipy.misc.imresize(image, size))
    return resizedImages


def labelsToOutputs(classes, labels):
    outputs = []
    for label in labels:
        zeros = [0] * len(classes)
        zeros[classes.index(int(label))] = 1
        outputs.append(zeros)
    return outputs

def createMapping(classes):
    return {idx: val for idx, val in enumerate(sorted(classes))}
