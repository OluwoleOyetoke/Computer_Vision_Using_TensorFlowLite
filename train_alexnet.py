
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os

tf.logging.set_verbosity(tf.logging.INFO)


#CREATE CNN STRUCTURE
def cnn_model_fn(features, labels, mode):

    #Input layer
    inpt_layer = tf.reshape(features["x"], [-1, 227, 227, 3])


    #Convolutional layer 1
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=96, kernel_size=[11, 11, 3], strides=4

    


if __name__=="__main__":
    tf.app.run()
