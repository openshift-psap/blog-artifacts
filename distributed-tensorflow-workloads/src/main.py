#!/usr/bin/python3

# TAKEN FROM: https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras

import os
import json

import tensorflow as tf

# Allow soft device placement
tf.config.set_soft_device_placement(True)

# Set autoshard policy to "DATA" since we're not reading from a file
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

import mnist

# The following code was taken from the link at the top of this file
###############################################################################
per_worker_batch_size = 64
tf_config = json.loads(os.environ['TF_CONFIG'])
num_workers = len(tf_config['cluster']['worker'])

strategy = tf.distribute.MultiWorkerMirroredStrategy()
global_batch_size = per_worker_batch_size * num_workers
multi_worker_dataset = mnist.mnist_dataset(global_batch_size)

with strategy.scope():
  # Model building/compiling need to be within `strategy.scope()`.
  multi_worker_model = mnist.build_and_compile_cnn_model()


multi_worker_model.fit(multi_worker_dataset, epochs=3, steps_per_epoch=700)
###############################################################################
