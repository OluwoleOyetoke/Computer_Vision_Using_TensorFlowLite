# Computer_Vision_Using_TensorFlowLite

On this project the AlexNet Convolution Neural Network is trained using traffic sign images from the German Road Traffic Sign Benchmark.
The initially trained network is then quantized/optimized for deployment on mobile devices using TensorFlowLite 

## Project Steps

1. Download German Traffic Sign Benchmark training images [from here](http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip)
2. Convert from .ppm to .jpg format
3. Label db folder appropriately
4. Convert dataset to TFRecord
5. Create CNN (alexnet)
6. Train CNN using TFRecord data
7. Test & evaluate trained model
8. Quantize/Tune/Optimize trained model for mobile deployment
9. Test & evaluate tuned model

### Steps 1, 2 & 3
The **DatasetConverter** folder contains the java program written to go through the GTSBR dataset, rename all its folders using the class name of the sets of images the folders contain. It also converts the image file types from .ppm to .jpeg

### Step 4: Convert Dataset to TFRecord
With TensorFlow, we can store our whole dataset and all of its meta-data as a serialized record called TFRecord. Loading up our data into this format helps for portability and simplicity. Also note that this record can be broken into multiple shards and used when performing distributed training. A 'TFRecord' in TensorFlow is basically TensorFlow's default data format. A record file containing serialized tf.train.examples. To avoid confusion, a TensorFlow 'example' is a normalized data format for storing data for training and inference purposes. It contains a key-value store where each key string maps to a feature message. These feature messages can be things like a packed byte list, float list, int64 list. Note that many 'examples' come together to form a TFRecord. The script **create_imdb.py** is used to make the TFRecords out of the dataset

### Step 5: Create AlexNet Structure
The script **train_alexnet.py** is used to create and train the CNN. See TensorBoard visualization of AlexNet structure


![Network Visualization](https://github.com/OluwoleOyetoke/Computer_Vision_Using_TensorFlowLite/blob/master/imgs/network_visualization.png)


### Step 6: Train CNN using TFRecord data
The figures below show the drop in loss of the network as training happened. The Adam Optimizer was used.
![Loss Per Epoch](https://github.com/OluwoleOyetoke/Computer_Vision_Using_TensorFlowLite/blob/master/imgs/loss_per_epoch.png)
![Accuracy Per Epoch](https://github.com/OluwoleOyetoke/Computer_Vision_Using_TensorFlowLite/blob/master/imgs/accuracy_per_epoch.png)

### Step 7: Test & evaluate trained model
To test the trained network model, the script **classify_img_arg.py** can be used.
Usage format:
'''python
$python one_time_classify.py (saved_model_directory) (path_to_image)
'''
