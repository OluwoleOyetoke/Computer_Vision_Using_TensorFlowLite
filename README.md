# Computer_Vision_Using_TensorFlowLite

On this project the AlexNet Convolutional Neural Network is trained using traffic sign images from the German Road Traffic Sign Benchmark. The initially trained network is then quantized/optimized for deployment on mobile devices using TensorFlowLite 

## Project Steps
1. Download German Traffic Sign Benchmark Training Images [from here](http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip)
2. Convert From .ppm to .jpg Format
3. Label db Folder Appropriately
4. Convert Dataset to TFRecord
5. Create CNN (alexnet)
6. Train CNN Using TFRecord Data
7. Test & Evaluate Trained Model
8. Quantize/Tune/Optimize trained model for mobile deployment
9. Test & Evaluate Tuned Model

### Steps 1, 2 & 3: Get Dataset, Convert (from .ppm to .jpeg) & Label Appropriately
The **DatasetConverter** folder contains the java program written to go through the GTSBR dataset, rename all its folders using the class name of the sets of images the folders contain. It also converts the image file types from .ppm to .jpeg

### Step 4: Convert Dataset to TFRecord
With TensorFlow, we can store our whole dataset and all of its meta-data as a serialized record called **TFRecord**. Loading up our data into this format helps for portability and simplicity. Also note that this record can be broken into multiple shards and used when performing distributed training. A 'TFRecord' in TensorFlow is basically TensorFlow's default data format. A record file containing serialized tf.train.examples. To avoid confusion, a TensorFlow 'example' is a normalized data format for storing data for training and inference purposes. It contains a key-value store where each key string maps to a feature message. These feature messages can be things like a packed byte list, float list, int64 list. Note that many 'examples' come together to form a TFRecord. The script **create_imdb.py** is used to make the TFRecords out of the dataset. To learn more about creating a TFRecord file or streaming out data from it, see these posts [TensorFlow TFRecord Blog Post 1](http://eagle-beacon.com/blog/posts/Loading_And_Poping_TFRecords.html) here & [TensorFlow TFRecord Blog Post 2](http://eagle-beacon.com/blog/posts/Loading_And_Poping_TFRecords_P2.html)

### Step 5: Create AlexNet Structure
The script **train_alexnet.py** is used to create and train the CNN. See TensorBoard visualization of AlexNet structure below:


![Network Visualization](https://github.com/OluwoleOyetoke/Computer_Vision_Using_TensorFlowLite/blob/master/imgs/network_visualization.png)

**Figure Showing TesnorBoard Visualization of the Network**


### Step 6: Train CNN Using TFRecord Data
The figures below show the drop in loss of the network as training progressed. The Adam Optimizer was used. The Figure below shows the loss reducing per epoch
![Loss Per Epoch](https://github.com/OluwoleOyetoke/Computer_Vision_Using_TensorFlowLite/blob/master/imgs/accuracy_per_epoch.png)

**Figure Showing Improvement In Network Performance Per Epoch (Epoch 0-20)**

After the full training procedure, the trained model performed at **over 98% accuracy**

### Step 7: Test & Evaluate Trained Model
To test the trained network model, the script **classify_img_arg.py** can be used.
Usage format:

```

$python one_time_classify.py [saved_model_directory] [path_to_image]

```

### Step 8: Quantize/Tune/Optimize Trained Model for Mobile Deployment
To convert the Tensorflow .pb model to a TensorFlowLite .lite model, we will:
1. Freeze the grpah i.e merge checkpoint values with graph stucture. In other words, load variables into the graph and convert them to constants
2. Convert the frozen graph definition into the the [flat buffer format](https://google.github.io/flatbuffers/) (.lite)
The TensorFlow graph optimization framework offers a suite of tools for modifying computational graphs.
Code excerpt below from the [**freeze_and_convert_to_tflite.sh**](https://github.com/OluwoleOyetoke/Computer_Vision_Using_TensorFlowLite/blob/master/freeze_and_convert_to_tflite.sh) script shows how this .tflite conversion is achieved. 

```
#FREEZE GRAPH
bazel build tensorflow/python/tools:freeze_graph
bazel-bin/tensorflow/python/tools/freeze_graph --input_graph=$1 --input_checkpoint=$2 --input_binary=$3 --output_graph=$4 --output_node_names=$6


#VIEW SUMARY OF GRAPH
bazel build /tensorflow/tools/graph_transforms:summarize_graph
bazel-bin/tensorflow/tools/graph_transforms/summarize_graph --in_graph=$4

#OPTIMIZE GRAPH
bazel build tensorflow/tools/graph_transforms:transform_graph
bazel-bin/tensorflow/tools/graph_transforms/transform_graph --in_graph=$4 --out_graph=${10} --inputs=$5 --outputs=$6 --transforms='strip_unused_nodes(type=float, shape="1,227,227,3") remove_nodes(op=Identity, op=CheckNumerics) fold_old_batch_norms fold_batch_norms'


#CONVERT TO TFLITE MODEL
bazel build /tensorflow/contrib/lite/toco:toco
bazel-bin/tensorflow/contrib/lite/toco/toco --input_format=TENSORFLOW_GRAPHDEF --input_file=${10} --output_format=TFLITE --output_file=$7 --inference_type=$9 --#input_type=$8 --input_arrays=$5 --output_arrays=$6 --inference_input_type=$8 --input_shapes=1,227,227,3

```

Note, this will require you to build TensorFlow from source. You can get instructions to do this from [here](https://www.tensorflow.org/install/install_sources)
