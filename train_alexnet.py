"""
    Used to Create AlexNet and Train it using image data stored in a TF Record over several Epochs

            @date: 31st March, 2017
            @author: Oluwole Oyetoke Jnr
            @Language: Python

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



#IMPORTS & VARIABLE DECLARATION
-------------------------------------------------------------------------------------------------------------------------------------------------------------------"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import numpy as np
import tensorflow as tf #Import tensorflow

#global NUM_OF_CLASSES = 43

"""----------------------------------------------------------------------------------------------------------------------------------------------------------------"""
tf.logging.set_verbosity(tf.logging.INFO) #Setting up logging


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


#CREATE CNN STRUCTURE
"""----------------------------------------------------------------------------------------------------------------------------------------------------------------"""
def cnn_model_fn(features, labels, mode):

    """iNPUT LAYER"""
    input_layer = tf.reshape(features["x"], [-1, 227, 227, 3]) #Alexnet uses 227x227x3 input layer. '-1' means pick batch size randomly
    print(input_layer)

    """%FIRST CONVOLUTION BLOCK
        The first convolutional layer filters the 227×227×3 input image with
        96 kernels of size 11×11×3 with a stride of 4 pixels. Bias of 1."""
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=96, kernel_size=[11, 11], strides=4, padding="valid", activation=tf.nn.relu)
    #Normalizaition layer
    lrn1 = tf.nn.lrn(input=conv1, depth_radius=5, bias=1.0, alpha=0.0001/5.0, beta=0.75);
    #Max Pool Layer
    pool1_conv1 = tf.layers.max_pooling2d(inputs=lrn1, pool_size=[3, 3], strides=2)
    print(pool1_conv1)
    print("end of conv block 1")
    

    """SECOND CONVOLUTION BLOCK
    Divide the 96 channel blob input from block one into 48 and process independently"""
    conv2 = tf.layers.conv2d(inputs=pool1_conv1, filters=256, kernel_size=[5, 5], strides=1, padding="same", activation=tf.nn.relu)
    #Normalizaition layer
    lrn2 = tf.nn.lrn(input=conv2, depth_radius=5, bias=1.0, alpha=0.0001/5.0, beta=0.75);
    #Max Pool Layer
    pool2_conv2 = tf.layers.max_pooling2d(inputs=lrn2, pool_size=[3, 3], strides=2) 
    print(pool2_conv2)
    print("end of conv block 2")

    """THIRD CONVOLUTION BLOCK
    Note that the The third, fourth, and fifth convolutioN layers are connected to one
    another without any intervening pooling or normalization layers.
    The third convolutional layer has 384 kernels of size 3 × 3 × 256
    connected to the (normalized, pooled) outputs of the second convolutional layer"""
    conv3 = tf.layers.conv2d(inputs=pool2_conv2, filters=384, kernel_size=[3, 3], strides=1, padding="same", activation=tf.nn.relu)
    print(conv3)
    print("end of conv block 3")

    #FOURTH CONVOLUTION BLOCK
    """%The fourth convolutional layer has 384 kernels of size 3 × 3 × 192"""
    conv4 = tf.layers.conv2d(inputs=conv3, filters=384, kernel_size=[3, 3], strides=1, padding="same", activation=tf.nn.relu)
    print(conv4)
    print("end of conv block 4")

    #FIFTH CONVOLUTION BLOCK
    """%the fifth convolutional layer has 256 kernels of size 3 × 3 × 192"""
    conv5 = tf.layers.conv2d(inputs=conv4, filters=256, kernel_size=[3, 3], strides=1, padding="same", activation=tf.nn.relu)
    print(conv5)
    #Max Pool Layer
    pool3_conv5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[3, 3], strides=2, padding="valid")
    print(pool3_conv5)
    print("end of conv block 5")

    pool3_conv5_flat = tf.reshape(pool3_conv5, [-1, 6* 6 * 256])
    
    #FULLY CONNECTED LAYER 1
    """The fully-connected layers have 4096 neurons each"""
    fc1 = tf.layers.dense(inputs=pool3_conv5_flat, units=4096, activation=tf.nn.relu)
    #fc1 = tf.layers.conv2d(inputs=pool3_conv5, filters=4096, kernel_size=[6, 6], strides=1, padding="valid", activation=tf.nn.relu)
    print(fc1)
    print("end of fc1")

    #FULLY CONNECTED LAYER 2
    """since the output from above is [1x1x4096]"""
    fc2 = tf.layers.dense(inputs=fc1, units=4096, activation=tf.nn.relu)
    #fc2 = tf.layers.conv2d(inputs=fc1, filters=4096, kernel_size=[1, 1], strides=1, padding="valid", activation=tf.nn.relu)
    print(fc2)
    print("end of fc2")

    #FULLY CONNECTED LAYER 3
    """since the output from above is [1x1x4096]"""
    logits = tf.layers.dense(inputs=fc2, units=43)
    #fc3 = tf.layers.conv2d(inputs=fc2, filters=43, kernel_size=[1, 1], strides=1, padding="valid")
    print(logits)
    print("end of fc3")
    #print(tf.shape(logits))

    #logits = tf.layers.dense(inputs=fc3, units=43)
    

    #PASS OUTPUT OF LAST FC LAYER TO A SOFTMAX LAYER
    """convert these raw values into two different formats that our model function can return:
    The predicted class for each example: a digit from 1–42.
    The probabilities for each possible target class for each example
    tf.argmax(input=fc3, axis=1: Generate predictions from the 43 last filters returned from the fc3
    tf.nn.softmax(fc3, name="softmax_tensor"): Generate the probability distribution
    """
    predictions = {
      "classes": tf.argmax(input=logits, axis=1),
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
      }
    print("end of prediction setting")
    #Return result if we were in prediction mode and not training
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    #CALCULATE OUR LOSS
    """For both training and evaluation, we need to define a loss function that measures how closely the
    model's predictions match the target classes. For multiclass classification, cross entropy is typically used as the loss metric."""
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=43)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
    print("end of loss calculation")
    
    #CONFIGURE TRAINING
    """Since the loss of the CNN is the softmax cross-entropy of the fc3 layer
    and our labels. Let's configure our model to optimize this loss value during
    training. We'll use a learning rate of 0.001 and stochastic gradient descent
    as the optimization algorithm:"""
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00005) #Very small learning rate used. Training will be slower at converging by better
        train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    print("end of optimization")
    #ADD EVALUATION METRICS
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

"""-----------------------------------------------------------------------------------------------------------------------------------------------------------------"""


#FUNCTION TO PROCESS ALL DATASET DATA
def _process_dataset(serialized): 
  #Specify the fatures you want to extract
  features = {'image/shape': tf.FixedLenFeature([], tf.string),
                'image/class/label': tf.FixedLenFeature([], tf.int64),
                'image/class/text': tf.FixedLenFeature([], tf.string),
                'image/filename': tf.FixedLenFeature([], tf.string),
                'image/encoded': tf.FixedLenFeature([], tf.string)} 
  parsed_example = tf.parse_single_example(serialized, features=features)

  #Finese extracted data
  image_raw = tf.decode_raw(parsed_example['image/encoded'], tf.uint8)
  shape = tf.decode_raw(parsed_example['image/shape'], tf.int32)
  label = tf.cast(parsed_example['image/class/label'], dtype=tf.int32)
  reshaped_img = tf.reshape(image_raw, [227, 227, 3])
  casted_img =  tf.cast(reshaped_img, tf.float32)
  label_tensor= [label]
  image_tensor = [casted_img]
  return label_tensor, image_tensor 

   

#TRAINING AND EVALUATING THE ALEXNET CNN CLASSIFIER
def main(unused_argv):

    #Declare needed variables
    perform_shuffle=False
    repeat_count=1
    dataset_batch_size=20
    num_of_epochs=1
    
    #LOAD TRAINING DATA
    print("DATSET LOADING STARTED\n\n")
    #filenames = ["/home/olu/Dev/data_base/sign_base/output/TFRecord_227x227/train-00000-of-00002", "/home/olu/Dev/data_base/sign_base/output/TFRecord_227x227/train-00001-of-00002"] #Directory path to the '.tfrecord' files
    filenames = ["C:/Users/Oluwole_Jnr/Desktop/Mobile Accelerated Vision App Project/TFRecord_227x227/train-00001-of-00002"]
    
    #Determine total number of records in the '.tfrecord' files
    print("GETING COUNT OF RECORDS/EXAMPLES IN DATASET")
    record_count = 19000
    #for fn in filenames:
      #for record in tf.python_io.tf_record_iterator(fn):
         #record_count += 1
    print("Total Number of Records in the .tfrecord file(s): %i\n\n" % record_count)
    

    print("CREATING ESTIMATOR AND LOADING DATASET")
    #CREATE ESTIMATOR
    """Estimator: a TensorFlow class for performing high-level model training, evaluation, and inference"""
    dataset_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="C:/Users/Oluwole_Jnr/Desktop/Mobile Accelerated Vision App Project/Dump/trained_alexnet_model") #Specified where the finally trained model should be saved in

    #SET-UP LOGGIN FOR PREDICTIONS
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)


    dataset = tf.data.TFRecordDataset(filenames=filenames)
    dataset = dataset.map(_process_dataset)                     #Get all content of dataset
    dataset = dataset.repeat(repeat_count)                      #Repeats dataset this # times
    dataset = dataset.batch(dataset_batch_size)                 #Batch size to use
    iterator = dataset.make_initializable_iterator()            #Create iterator which helps to get all iamges in the dataset
    labels_tensor, images_tensor = iterator.get_next()          #Get batch data
    no_of_rounds = int(math.ceil(record_count/dataset_batch_size));

    #Create tf session, get nest set of batches, and evelauate them in batches
    sess = tf.Session()
    print("Total number of strides needed to stream through dataset: ~%i" %no_of_rounds) 
  
    count=1
    for _ in range(num_of_epochs):
      sess.run(iterator.initializer)
      
      while True:
        try:
          print("Now evaluating tensors for stride %i out of %i and feedin it into classifer for training" % (count, no_of_rounds))
          evaluated_label, evaluated_image = sess.run([labels_tensor, images_tensor])
          #convert evaluated tensors to np array 
          label_np_array = np.asarray(evaluated_label, dtype=np.int32)
          image_np_array = np.asarray(evaluated_image, dtype=np.float32)
          #squeeze np array to make dimesnsions appropriate
          squeezed_label_np_array = label_np_array.squeeze()
          squeezed_image_np_array = image_np_array.squeeze()
          #Feed current batch to TF Estimator for training
          train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": squeezed_image_np_array},y=squeezed_label_np_array,batch_size=10,num_epochs=1, shuffle=True)
          dataset_classifier.train(input_fn=train_input_fn,hooks=[logging_hook])
        except tf.errors.OutOfRangeError:
          print("End of Dataset Reached")
          break
        count=count+1
    sess.close()

    print("End of Training")
    
    #EVALUATE MODEL
    """Once trainingis completed, we then proceed to evaluate the accuracy level of our trained model
    To create eval_input_fn, we set num_epochs=1, so that the model evaluates the metrics over one epoch of
    data and returns the result. We also set shuffle=False to iterate through the data sequentially."""
    #eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data},y=eval_labels,num_epochs=1,shuffle=False)
    #eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    #print(eval_results)
    
if __name__=="__main__":
    tf.app.run()
