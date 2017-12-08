"""

  AlexNet NETWORK OVERVIEW
AlexNet Structure: 60 million Parameters
8 layers in total: 5 Convolutional and 3 Fully Connected Layers
[227x227x3] INPUT
[55x55x96] CONV1: 96 11x11 filters at stride 4, pad 0
[27x27x96] MAX POOL1: 3x3 filters at stride 2
[27x27x96] NORM1: Normalization layer
[27x27x256] CONV2: 256 5x5 filters at stride 1, pad 2
[13x13x256] MAX POOL2: 3x3 filters at stride 2
[13x13x256] NORM2: Normalization layer
[13x13x384] CONV3: 384 3x3 filters at stride 1, pad 1
[13x13x384] CONV4: 384 3x3 filters at stride 1, pad 1
[13x13x256] CONV5: 256 3x3 filters at stride 1, pad 1
[6x6x256] MAX POOL3: 3x3 filters at stride 2
[4096] FC6: 4096 neurons
[4096] FC7: 4096 neurons
[1000] FC8: 43 neurons (class scores)


"""


#IMPORTS
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
    input_layer = tf.reshape(features["x"], [-1, 227, 227, 3])


    #CONVOLUTIONAL BLOCK 1
    """%FIRST CONVOLUTIONAL BLOCK
        The first convolutional layer filters the 227×227×3 input image with
        96 kernels of size 11×11×3 with a stride of 4 pixels. Bias of 1."""
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=96, kernel_size=[11, 11, 3], strides=4, padding="valid", activation=tf.nn.relu)
    #Normalizaition layer
    lrn1 = tf.nn.lrn(inputs=conv1);
    #Max Pool Layer
    pool1_conv1 = tf.layers.max_pooling2d(inputs=lrn1, pool_size=[3, 3], strides=2)



    #SECOND CONVOLUTIONAL BLOCK
    """Divide the 96 channel blob input from block one into 48 and process independently"""
    paddings = tf.constant([[1, 2,], [2, 2]])
    padded = tf.pad(pool1_conv1, paddings, "CONSTANT")
    conv2 = tf.layers.conv2d(inputs=padded, filters=256, kernel_size=[5, 5, 48], strides=4, padding="valid", activation=tf.nn.relu)
    lrn1 = tf.nn.lrn(inputs=conv2 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75);

if __name__=="__main__":
    tf.app.run()
"""

  AlexNet NETWORK OVERVIEW
AlexNet Structure: 60 million Parameters
8 layers in total: 5 Convolutional and 3 Fully Connected Layers
[227x227x3] INPUT
[55x55x96] CONV1: 96 11x11 filters at stride 4, pad 0
[27x27x96] MAX POOL1: 3x3 filters at stride 2
[27x27x96] NORM1: Normalization layer
[27x27x256] CONV2: 256 5x5 filters at stride 1, pad 2
[13x13x256] MAX POOL2: 3x3 filters at stride 2
[13x13x256] NORM2: Normalization layer
[13x13x384] CONV3: 384 3x3 filters at stride 1, pad 1
[13x13x384] CONV4: 384 3x3 filters at stride 1, pad 1
[13x13x256] CONV5: 256 3x3 filters at stride 1, pad 1
[6x6x256] MAX POOL3: 3x3 filters at stride 2
[4096] FC6: 4096 neurons
[4096] FC7: 4096 neurons
[1000] FC8: 43 neurons (class scores)


"""


#IMPORTS
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
    input_layer = tf.reshape(features["x"], [-1, 227, 227, 3])


    #CONVOLUTIONAL BLOCK 1
    """%FIRST CONVOLUTIONAL BLOCK
        The first convolutional layer filters the 227×227×3 input image with
        96 kernels of size 11×11×3 with a stride of 4 pixels. Bias of 1."""
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=96, kernel_size=[11, 11, 3], strides=4, padding="valid", activation=tf.nn.relu)
    #Normalizaition layer
    lrn1 = tf.nn.lrn(inputs=conv1);
    #Max Pool Layer
    pool1_conv1 = tf.layers.max_pooling2d(inputs=lrn1, pool_size=[3, 3], strides=2)



    #SECOND CONVOLUTIONAL BLOCK
    """Divide the 96 channel blob input from block one into 48 and process independently"""
    paddings = tf.constant([[1, 2,], [2, 2]])
    padded = tf.pad(pool1_conv1, paddings, "CONSTANT")
    conv2 = tf.layers.conv2d(inputs=padded, filters=256, kernel_size=[5, 5, 48], strides=4, padding="valid", activation=tf.nn.relu)
    lrn1 = tf.nn.lrn(inputs=conv2 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75);

if __name__=="__main__":
    tf.app.run()
