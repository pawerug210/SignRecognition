import scipy.misc
import numpy as np
import tensorflow as tf


def crop(image, startX, stopX, startY, stopY):
    return image[startX: stopX + 1, startY: stopY + 1, :]  # slice end border inclusive ( + 1 )


def resizeTF(images, size):
    images = tf.convert_to_tensor(images)
    return tf.image.resize_images(images, size)


def resize(images, size):
    # todo yield
    resizedImages = []
    for image in images:
        resizedImages.append(scipy.misc.imresize(image, size))
    return resizedImages