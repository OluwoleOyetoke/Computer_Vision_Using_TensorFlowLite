#!/bin/bash

#################################################################################################################
# Bash script to help freez tensorflow GraphDef + ckpt into FrozenGraphDef					#
# $1 = path/to/.pbtx file 											#
# $2 = path/to/.ckpt file											#
# $3 = true/false. 'true' for .pb and 'false' for .pbtx
# $4 = path/to/save/frozengraph											#
# $5 = output_node_name												#
# Sample usage:													#
# freez_graph_tf.sh /tmp/graph.pbtx /tmp/model.ckpt-0.data-00000-of-00001 false /tmp/frozen.pb ArgMax													#
#														#
#														#
#														#
#################################################################################################################



#build original TensorFlow freeze_graph script
bazel build /home/olu/TensorFlow_Source/tensorflow/bazel-bin/tensorflow/python/tools:freeze_graph

/home/olu/TensorFlow_Source/tensorflow/bazel-bin/tensorflow/python/tools/freeze_graph --input_graph=$1 --input_checkpoint=$2 --input_binary=$3 --output_graph=$4 --output_node_names=$5

