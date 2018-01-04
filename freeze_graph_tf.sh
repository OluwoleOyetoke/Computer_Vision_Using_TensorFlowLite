#!/bin/bash

#################################################################################################################
# Bash script to help freez tensorflow GraphDef + ckpt into FrozenGraphDef					#
# $1 = path/to/.pbtx file 											#
# $2 = path/to/.ckpt file											#
# $3 = true/false. 'true' for .pb and 'false' for .pbtx input
# $4 = path/to/save/frozengraph	
# $5 = graph input node name										#
# $6 = graph output_node_name		
# $7 = path/to/tflite/file e.g path/to/tflite_model.lite
# $8 = input type e.g FLOAT# 
# $9 = inference (output) type (FLOAT or QUANTIZED)
# 
#   												                #
# Sample usage:													#
# freez_graph_tf.sh /tmp/graph.pbtx /tmp/model.ckpt-0 false /tmp/frozen.pb ArgMax				#
#														#
#														#
#														#
#################################################################################################################


#Location of freeze_graph for you may be different
#python /home/olu/.local/lib/python2.7/site-packages/tensorflow/python/tools/freeze_graph.py --input_graph=$1 --input_checkpoint=$2 --input_binary=$3 --output_graph=$4 --output_node_names=$6

bazel build /home/olu/TensorFlow_Source/tensorflow/bazel-bin/tensorflow/contrib/lite/toco:toco
/home/olu/TensorFlow_Source/tensorflow/bazel-bin/tensorflow/contrib/lite/toco/toco --input_format=TENSORFLOW_GRAPHDEF --input_file=$4 --output_format=TFLITE --output_file=$7 --inference_type=$9 --input_type=$8 --input_arrays=$5 --output_arrays=$6 --inference_input_type=$8 --input_shapes=1,227,227,3

#Convert to lite model
#/home/olu/.local/lib/python2.7/site-packages/tensorflow/contrib/lite/toco --input_file=$4 --input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE \
#  --output_file=$7 --inference_type=$9 --input_type=$8 --input_arrays=$5 --output_arrays=$6 --input_shapes=1,227,227,3
