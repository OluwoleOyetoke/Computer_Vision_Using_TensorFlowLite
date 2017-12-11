

"""
    Used to Create AlexNet and Train it using image data stored in a TF Record over several Epochs

            @date: 4th December, 2017
            @author: Oluwole Oyetoke
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






#IMPORTS, VARIABLE DECLARATION, AND LOGGING TYPE SETTING
----------------------------------------------------------------------------------------------------------------------------------------------------------------"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn.model_selection import train_test_split

import os
import math
import numpy as np
import tensorflow as tf #import tensorflow

flags = tf.app.flags
flags.DEFINE_integer("image_width", "227", "Alexnet input layer width")
flags.DEFINE_integer("image_height", "227", "Alexnet input layer height")
flags.DEFINE_integer("image_channels", "3", "Alexnet input layer channels")
flags.DEFINE_integer("num_of_classes", "43", "Number of training clases")
FLAGS = flags.FLAGS

losses_bank = np.array([]) #global

tf.logging.set_verbosity(tf.logging.INFO) #setting up logging (can be DEBUG, ERROR, FATAL, INFO or WARN)
"""----------------------------------------------------------------------------------------------------------------------------------------------------------------"""






#WRAPPER FOR INSERTING int64 FEATURES int64 FEATURES & BYTES FEATURES INTO EXAMPLES PROTO
"""----------------------------------------------------------------------------------------------------------------------------------------------------------------"""
def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
"""----------------------------------------------------------------------------------------------------------------------------------------------------------------"""




#CREATE CNN STRUCTURE
"""----------------------------------------------------------------------------------------------------------------------------------------------------------------"""
def cnn_model_fn(features, labels, mode):

    """INPUT LAYER"""
    input_layer = tf.reshape(features["x"], [-1, FLAGS.image_width, FLAGS.image_height, FLAGS.image_channels]) #Alexnet uses 227x227x3 input layer. '-1' means pick batch size randomly
    print(input_layer)

    """%FIRST CONVOLUTION BLOCK
        The first convolutional layer filters the 227×227×3 input image with
        96 kernels of size 11×11 with a stride of 4 pixels. Bias of 1."""
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=96, kernel_size=[11, 11], strides=4, padding="valid", activation=tf.nn.relu)
    lrn1 = tf.nn.lrn(input=conv1, depth_radius=5, bias=1.0, alpha=0.0001/5.0, beta=0.75); #Normalization layer
    pool1_conv1 = tf.layers.max_pooling2d(inputs=lrn1, pool_size=[3, 3], strides=2) #Max Pool Layer
    print(pool1_conv1)
    print("end of conv block 1")
    

    """SECOND CONVOLUTION BLOCK
    Divide the 96 channel blob input from block one into 48 and process independently"""
    conv2 = tf.layers.conv2d(inputs=pool1_conv1, filters=256, kernel_size=[5, 5], strides=1, padding="same", activation=tf.nn.relu)
    lrn2 = tf.nn.lrn(input=conv2, depth_radius=5, bias=1.0, alpha=0.0001/5.0, beta=0.75); #Normalization layer
    pool2_conv2 = tf.layers.max_pooling2d(inputs=lrn2, pool_size=[3, 3], strides=2) #Max Pool Layer
    print(pool2_conv2)
    print("end of conv block 2")

    """THIRD CONVOLUTION BLOCK
    Note that the third, fourth, and fifth convolution layers are connected to one
    another without any intervening pooling or normalization layers.
    The third convolutional layer has 384 kernels of size 3 × 3
    connected to the (normalized, pooled) outputs of the second convolutional layer"""
    conv3 = tf.layers.conv2d(inputs=pool2_conv2, filters=384, kernel_size=[3, 3], strides=1, padding="same", activation=tf.nn.relu)
    print(conv3)
    print("end of conv block 3")

    #FOURTH CONVOLUTION BLOCK
    """%The fourth convolutional layer has 384 kernels of size 3 × 3"""
    conv4 = tf.layers.conv2d(inputs=conv3, filters=384, kernel_size=[3, 3], strides=1, padding="same", activation=tf.nn.relu)
    print(conv4)
    print("end of conv block 4")

    #FIFTH CONVOLUTION BLOCK
    """%the fifth convolutional layer has 256 kernels of size 3 × 3"""
    conv5 = tf.layers.conv2d(inputs=conv4, filters=256, kernel_size=[3, 3], strides=1, padding="same", activation=tf.nn.relu)
    pool3_conv5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[3, 3], strides=2, padding="valid") #Max Pool Layer
    print(pool3_conv5)
    print("end of conv block 5")


    #FULLY CONNECTED LAYER 1
    """The fully-connected layers have 4096 neurons each"""
    pool3_conv5_flat = tf.reshape(pool3_conv5, [-1, 6* 6 * 256]) #output of conv block is 6x6x256 therefore, to connect it to a fully connected layer, we can flaten it out
    fc1 = tf.layers.dense(inputs=pool3_conv5_flat, units=4096, activation=tf.nn.relu)
    #fc1 = tf.layers.conv2d(inputs=pool3_conv5, filters=4096, kernel_size=[6, 6], strides=1, padding="valid", activation=tf.nn.relu) #representing the FCL using a convolution block (no need to do 'pool3_conv5_flat' above)
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
    logits = tf.layers.dense(inputs=fc2, units=FLAGS.num_of_classes)
    #fc3 = tf.layers.conv2d(inputs=fc2, filters=43, kernel_size=[1, 1], strides=1, padding="valid")
    #logits = tf.layers.dense(inputs=fc3, units=FLAGS.num_of_classes) #converting the convolutional block (tf.layers.conv2d) to a dense layer (tf.layers.dense). Only needed if we had used tf.layers.conv2d to represent the FCLs
    print(logits)
    print("end of fc3")

    #PASS OUTPUT OF LAST FC LAYER TO A SOFTMAX LAYER
    """convert these raw values into two different formats that our model function can return:
    The predicted class for each example: a digit from 1–43.
    The probabilities for each possible target class for each example
    tf.argmax(input=fc3, axis=1: Generate predictions from the 43 last filters returned from the fc3
    tf.nn.softmax(logits, name="softmax_tensor"): Generate the probability distribution
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
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=FLAGS.num_of_classes)
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
    print("end of evaluation")
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
"""-----------------------------------------------------------------------------------------------------------------------------------------------------------------"""




#FUNCTION TO PROCESS ALL DATASET DATA
"""----------------------------------------------------------------------------------------------------------------------------------------------------------------"""
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
  reshaped_img = tf.reshape(image_raw, [FLAGS.image_width, FLAGS.image_height, FLAGS.image_channels])
  casted_img =  tf.cast(reshaped_img, tf.float32)
  label_tensor= [label]
  image_tensor = [casted_img]
  return label_tensor, image_tensor 
"""----------------------------------------------------------------------------------------------------------------------------------------------------------------"""
   
#PLOT TRAINING PROGRESS
def _plot_training_progress():
  global losses_bank #to make sure losses_bank is not declared again in this method (as a local variable)


#TRAINING AND EVALUATING THE ALEXNET CNN CLASSIFIER
"""----------------------------------------------------------------------------------------------------------------------------------------------------------------"""
def main(unused_argv):

    #Declare needed variables
    perform_shuffle=False
    repeat_count=1
    dataset_batch_size=1024 #Chuncks picked in dataset per time
    training_batch_size = np.float32(dataset_batch_size/32) #Chuncks processed by tf.estimator per time
    epoch_count=0
    overall_training_epochs=1
    
    
    #LOAD TRAINING DATA
    print("DATSET LOADING STARTED\n\n")
    filenames = ["/home/olu/Dev/data_base/sign_base/output/TFRecord_227x227/train-00000-of-00002", "/home/olu/Dev/data_base/sign_base/output/TFRecord_227x227/train-00001-of-00002"] #Directory path to the '.tfrecord' files
    
    #DETERMINE TOTAL NUMBER OF RECORDS IN THE '.tfrecord' FILES
    print("GETING COUNT OF RECORDS/EXAMPLES IN DATASET")
    record_count = 0
    for fn in filenames:
      for record in tf.python_io.tf_record_iterator(fn):
         record_count += 1
    print("Total number of records in the .tfrecord file(s): %i\n\n" % record_count)
    

    print("CREATING ESTIMATOR AND LOADING DATASET")
    #CREATE ESTIMATOR
    """Estimator: a TensorFlow class for performing high-level model training, evaluation, and inference"""
    traffic_sign_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="/home/olu/Dev/data_base/sign_base/output/Checkpoints_N_Model/trained_alexnet_model") #SPECIFY where the finally trained model and (checkpoints during training) should be saved in

    #SET-UP LOGGIN FOR PREDICTIONS
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50) #Log after every 50 itterations

    #PROCESS AND RETREIVE DATASET CONTENT IN BATCHES OF 'dataset_batch_size'
    dataset = tf.data.TFRecordDataset(filenames=filenames)
    dataset = dataset.map(_process_dataset)                     #Get all content of dataset & apply function '_process_dataset' to all its content
    dataset = dataset.shuffle(buffer_size=1000)                 #Shuffle selection from the dataset/epoch
    dataset = dataset.repeat(repeat_count)                      #Repeat ittereation through dataset 'repeat_count' times
    dataset = dataset.batch(dataset_batch_size)                 #Batch size to use to pick from dataset
    iterator = dataset.make_initializable_iterator()            #Create iterator which helps to get all iamges in the dataset
    labels_tensor, images_tensor = iterator.get_next()          #Get batch data
    no_of_rounds = int(math.ceil(record_count/dataset_batch_size));

    #CREATE TF SESSION TO ITTERATIVELY EVALUATE THE BATCHES OF DATASET TENSORS RETREIVED AND PASS THEM TO ESTIMATOR FOR TRAINING/EVALUATION
    sess = tf.Session()
    print("Total number of strides needed to stream through 1 epoch of dataset: ~%i" %no_of_rounds) 
  
    
    for _ in range(overall_training_epochs):
      sess.run(iterator.initializer)
      epoch_count=epoch_count+1;
      strides_count=1
      complete_evaluation_image_set = np.array([])
      complete_evaluation_label_set = np.array([])
      while True:
        try:
          print("EPOCH %i out of %i: Now evaluating tensors for stride %i out of %i (%i images) and feeding it into classifer for training in batches of %i" % (epoch_count, overall_training_epochs, strides_count, no_of_rounds, dataset_batch_size, training_batch_size))
          print("Note: Each complete epoch processes %i images" % record_count)
          evaluated_label, evaluated_image = sess.run([labels_tensor, images_tensor])
          #convert evaluated tensors to np array 
          label_np_array = np.asarray(evaluated_label, dtype=np.int32)
          image_np_array = np.asarray(evaluated_image, dtype=np.float32)
          #squeeze np array to make dimesnsions appropriate
          squeezed_label_np_array = label_np_array.squeeze()
          squeezed_image_np_array = image_np_array.squeeze()
          #mean normalization - normalize current batch of images i.e get mean of images in dataset and subtact it from all image intensities in the dataset
          dataset_image_mean = np.mean(squeezed_image_np_array)
          normalized_image_dataset = np.subtract(squeezed_image_np_array, dataset_image_mean) #help for faster convergence duing training
          #Split data into training and testing/evaluation data
          image_train, image_evaluate, label_train, label_evaluate = train_test_split(normalized_image_dataset, squeezed_label_np_array, test_size=0.05, random_state=42, shuffle=True) #5% of dataset will be used for evaluation/testing
          normalized_image_dataset.astype(np.float32) #should be image_train
          #Store evaluation data in its place
          complete_evaluation_image_set = np.append(complete_evaluation_image_set, image_evaluate)
          complete_evaluation_label_set = np.append(complete_evaluation_label_set, label_evaluate)
          #Feed current batch of training images to TF Estimator for training. TF Estimator deals with them in batches of 'batch_size=32'
          train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": normalized_image_dataset},y=squeezed_label_np_array,batch_size=32,num_epochs=1, shuffle=True) #Note, images have already been shuffled when placed in the TFRecord, shuffled again when being retreived from the record & will be shuffled again when being sent to the classifier
          traffic_sign_classifier.train(input_fn=train_input_fn,hooks=[logging_hook])
        except tf.errors.OutOfRangeError:
          print("End of Dataset Reached")
          break
        strides_count=strides_count+1
      #EVALUATE MODEL (after every batche's epoch). Normally should be done after every complete epoch, but out of memory issues happen if all 20% of the dataset's images need to be stored in memory till full epoch is completed
      """Once trainingis completed, we then proceed to evaluate the accuracy level of our trained model
      To create eval_input_fn, we set num_epochs=1, so that the model evaluates the metrics over one epoch of
      data and returns the result. We also set shuffle=False to iterate through the data sequentially."""
      eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": complete_evaluation_image_set},y=complete_evaluation_label_set,num_epochs=1,shuffle=False)
      eval_results = traffic_sign_classifier.evaluate(input_fn=eval_input_fn)
      print(eval_results)
    sess.close()

    print("END OF TRAINING")
    
"""----------------------------------------------------------------------------------------------------------------------------------------------------------------"""


if __name__=="__main__":
    tf.app.run()
