# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Modifications copyright (C) 2017 Francesco Orabona, Tatiana Tommasi 

"""A fully connected MNIST classifier using convolutional layers.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import input_data

import tensorflow as tf
import numpy as np

sys.path.insert(0, '../optimizer/')

import cocob_optimizer

FLAGS = None


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def main(_):
  # Import data
  mnist = input_data.read_data_sets('data', one_hot=True, validation_size=0)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  W_fc1 = weight_variable([28*28, 1000])
  b_fc1 = bias_variable([1000])
  h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

  W_fc2 = weight_variable([1000, 1000])
  b_fc2 = bias_variable([1000])
  h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
  
  W_fc3 = weight_variable([1000, 10])
  b_fc3 = bias_variable([10])
  out = tf.matmul(h_fc2, W_fc3) + b_fc3
  
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=out))
  train_step = cocob_optimizer.COCOB().minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(600*40):
      batch = mnist.train.next_batch(100)
      if i % 600 == 0:
        test_batch_size = 10000
        batch_num = int(mnist.train.num_examples / test_batch_size)
        train_loss = 0
    
        for j in range(batch_num):
            train_loss += cross_entropy.eval(feed_dict={x: mnist.train.images[test_batch_size*j:test_batch_size*(j+1), :],
                                              y_: mnist.train.labels[test_batch_size*j:test_batch_size*(j+1), :]})
            
        train_loss /= batch_num

        test_err = 1-accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})

        print('epoch %d, training cost %g, test error %g ' % (i/600, train_loss, test_err))
      train_step.run(feed_dict={x: batch[0], y_: batch[1]})

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
