#!/bin/bash

#################################################################################################################
# Bash script to help freez tensorflow GraphDef + ckpt into FrozenGraphDef, Optimize and convert to tflite model#
# $1 = path/to/.pbtx file 											#
# $2 = path/to/.ckpt file											#
# $3 = true/false. 'true' for .pb and 'false' for .pbtx input							#
# $4 = path/to/save/frozengraph											#
# $5 = graph input node name											#
# $6 = graph output_node_name											#
# $7 = path/to/tflite/file e.g path/to/tflite_model.lite							#
# $8 = input type e.g FLOAT											# 			
# $9 = inference (output) type (FLOAT or QUANTIZED)								#
# $10 = path/to/optimized_graph.pb										#
#   												                #
# Sample usage:													#
# freez_graph_tf.sh /tmp/graph.pbtx /tmp/model.ckpt-0 false /tmp/frozen.pb ArgMax				#
#														#
#	Make sure you are runing this from the tensorflow home folder after cloning the TF repository		#
#														#
#################################################################################################################


#FREEZE GRAPH
#bazel build tensorflow/python/tools:freeze_graph
bazel-bin/tensorflow/python/tools/freeze_graph --input_graph=$1 --input_checkpoint=$2 --input_binary=$3 --output_graph=$4 --output_node_names=$6


#VIEW SUMARY OF GRAPH
#bazel build /tensorflow/tools/graph_transforms:summarize_graph
bazel-bin/tensorflow/tools/graph_transforms/summarize_graph --in_graph=$4

#OPTIMIZE GRAPH
#bazel build tensorflow/tools/graph_transforms:transform_graph
bazel-bin/tensorflow/tools/graph_transforms/transform_graph --in_graph=$4 --out_graph=${10} --inputs=$5 --outputs=$6 --transforms='strip_unused_nodes(type=float, shape="1,227,227,3") remove_nodes(op=Identity, op=CheckNumerics) fold_old_batch_norms fold_batch_norms'


#CONVERT TO TFLITE MODEL
#bazel build /tensorflow/contrib/lite/toco:toco
bazel-bin/tensorflow/contrib/lite/toco/toco --input_format=TENSORFLOW_GRAPHDEF --input_file=${10} --output_format=TFLITE --output_file=$7 --inference_type=$9 --#input_type=$8 --input_arrays=$5 --output_arrays=$6 --inference_input_type=$8 --input_shapes=1,227,227,3









