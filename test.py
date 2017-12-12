#IMPORTS
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn.model_selection import train_test_split

#import Image
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import time

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


#FUNCTION TO GET ALL DATASET DATA
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
  reshaped_img = tf.reshape(image_raw, shape)
  casted_img =  tf.cast(reshaped_img, tf.float32)
  label_tensor= [label]
  image_tensor = [casted_img]
  return label_tensor, image_tensor 


#MAIN FUNCTION
def main(unused_argv):

    
    print("STARTED\n\n")

    #Declare needed variables
    perform_shuffle=False
    repeat_count=1
    batch_size=1000
    num_of_epochs=1
    
    #Directory path to the '.tfrecord' files
    filenames = ["/home/olu/Dev/data_base/sign_base/output/TFRecord_227x227/train-00000-of-00002", "/home/olu/Dev/data_base/sign_base/output/TFRecord_227x227/train-00001-of-00002"]


    print("GETTING RECORD COUNT")
    #Determine total number of records in the '.tfrecord' files
    record_count = 0
    for fn in filenames:
      for record in tf.python_io.tf_record_iterator(fn):
         record_count += 1
    print("Total Number of Records in the .tfrecord file(s): %i" % record_count)
  
 
    dataset = tf.data.TFRecordDataset(filenames=filenames)
    dataset = dataset.map(_process_dataset)                     #Get all content of dataset
    dataset = dataset.shuffle(buffer_size=1000)                 #Shuffle selection from the dataset
    dataset = dataset.repeat(repeat_count)                      #Repeats dataset this # times
    dataset = dataset.batch(batch_size)                         #Batch size to use
    iterator = dataset.make_initializable_iterator()            #Create iterator which helps to get all iamges in the dataset
    labels_tensor, images_tensor = iterator.get_next()          #Get batch data
    no_of_rounds = int(math.ceil(record_count/batch_size));

    #Create tf session, get nest set of batches, and evelauate them in batches
    sess = tf.Session()
    print("Total number of strides needed to stream through dataset: ~%i" %no_of_rounds) 
  
    
    for _ in range(2):
      sess.run(iterator.initializer)
      count=0
      complete_evaluation_image_set = np.array([])
      complete_evaluation_label_set = np.array([])
      while True:
        try:
          print("Now evaluating tensors for stride %i out of %i" % (count, no_of_rounds))
          evaluated_label, evaluated_image = sess.run([labels_tensor, images_tensor])
          #convert evaluated tensors to np array 
          label_np_array = np.asarray(evaluated_label, dtype=np.uint8)
          image_np_array = np.asarray(evaluated_image, dtype=np.uint8)
          #squeeze np array to make dimesnsions appropriate
          squeezed_label_np_array = label_np_array.squeeze()
          squeezed_image_np_array = image_np_array.squeeze()
          #Split data into training and testing data
          image_train, image_test, label_train, label_test = train_test_split(squeezed_image_np_array, squeezed_label_np_array, test_size=0.010, random_state=42, shuffle=True)
          #Store evaluation data in its place
          complete_evaluation_image_set = np.append(complete_evaluation_image_set, image_test)
          complete_evaluation_label_set = np.append(complete_evaluation_label_set, label_test)
          #Feed current batch to TF Estimator for training
        except tf.errors.OutOfRangeError:
          print("End of Dataset Reached")
          break
        count=count+1
      print(complete_evaluation_label_set.shape)
      print(complete_evaluation_image_set.shape)
    sess.close()

    print("End of Training")
    

    
if __name__ == "__main__":
    tf.app.run()

