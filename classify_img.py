"""
    Script passes raw image to an already trained model to get prediction

            @date: 20th December, 2017
            @author: Oluwole Oyetoke
            @Language: Python
            @email: oluwoleoyetoke@gmail.com

#IMPORTS, VARIABLE DECLARATION, AND LOGGING TYPE SETTING
----------------------------------------------------------------------------------------------------------------------------------------------------------------"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn.model_selection import train_test_split

import os
import math
import time
import numpy as np
import tensorflow as tf #import tensorflow
import matplotlib.pyplot as plt
from PIL import Image

flags = tf.app.flags
flags.DEFINE_integer("image_width", "227", "Alexnet input layer width")
flags.DEFINE_integer("image_height", "227", "Alexnet input layer height")
flags.DEFINE_integer("image_channels", "3", "Alexnet input layer channels")
flags.DEFINE_integer("num_of_classes", "43", "Number of training clases")
FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.WARN) #setting up logging (can be DEBUG, ERROR, FATAL, INFO or WARN)
"""----------------------------------------------------------------------------------------------------------------------------------------------------------------"""


#TRAINING AND EVALUATING THE ALEXNET CNN CLASSIFIER
"""----------------------------------------------------------------------------------------------------------------------------------------------------------------"""
def main(unused_argv):

  #Specify checkpoint & image directory
  checkpoint_directory="/home/olu/Dev/data_base/sign_base/backup/Checkpoints_N_Model_Epoch_34__copy/trained_alexnet_model"
  filename="/home/olu/Dev/data_base/sign_base/training_227x227/road_closed/00002_00005.jpeg"

  #Process image to be sent to Neural Net
  #im = np.array(Image.open(filename))
  #img_batch = im.reshape(1, FLAGS.image_width, FLAGS.image_height,FLAGS.image_channels)
  
  img = Image.open(filename)
  img_resized = img.resize((227, 227), Image.ANTIALIAS)
  img_batch_np = np.array(img_resized)
  img_batch = img_batch_np.reshape(1, FLAGS.image_width, FLAGS.image_height,FLAGS.image_channels)
  plt.imshow(img_batch_np)

  #Declare categories/classes as string
  categories = ["speed_20", "speed_30","speed_50","speed_60","speed_70",
    "speed_80","speed_less_80","speed_100","speed_120",
    "no_car_overtaking","no_truck_overtaking","priority_road",
    "priority_road_2","yield_right_of_way","stop","road_closed",
    "maximum_weight_allowed","entry_prohibited","danger","curve_left",
    "curve_right","double_curve_right","rough_road","slippery_road",
    "road_narrows_right","work_in_progress","traffic_light_ahead",
    "pedestrian_crosswalk","children_area","bicycle_crossing",
    "beware_of_ice","wild_animal_crossing","end_of_restriction",
    "must_turn_right","must_turn_left","must_go_straight",
    "must_go_straight_or_right","must_go_straight_or_left",
    "mandatroy_direction_bypass_obstacle",
    "mandatroy_direction_bypass_obstacle2", 
    "traffic_circle","end_of_no_car_overtaking",
    "end_of_no_truck_overtaking"];


  #Recreate network graph.  
  sess = tf.Session()
  latest_checkpoint_name = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_directory)
  saver = tf.train.import_meta_graph(latest_checkpoint_name+'.meta') #At this step only graph is created.
    
  #Accessing the default graph which we have restored
  graph = tf.get_default_graph()

  #Get model's graph
  checkpoint_file=tf.train.latest_checkpoint(checkpoint_directory)
  saver.restore(sess, checkpoint_file) #Load the weights saved using the restore method.

  probabilities = graph.get_tensor_by_name("softmax_tensor:0")
  classes = graph.get_tensor_by_name("classes_tensor:0") #'ArgMax:0' is the name of the argmax tensor in the train_alexnet.py file.
  feed_dict = {"input_layer:0": img_batch} #'Reshape:0' is the name of the 'input_layer' tensor in the train_alexnet.py. Given to it as default.
  predicted_class = sess.run(classes, feed_dict)
  predicted_probabilities = sess.run(probabilities, feed_dict)
  assurance = predicted_probabilities[0,int(predicted_class)]*100;      
      
  print("Predicted Sign: ", categories[int(predicted_class)], " (With ", assurance," Percent Assurance)")
  print("finished")
  plt.show()
"""----------------------------------------------------------------------------------------------------------------------------------------------------------------"""


if __name__=="__main__":
    tf.app.run()

