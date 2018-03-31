# MIT License
#
# Copyright (c) 2018 Blanyal D'Souza
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================
"""Class to represent the Neural Network."""

from config import CFG
import tensorflow as tf


class NeuralNetwork(object):
    """Represents the Policy and Value Resnet.

    Attributes:
        side: An integer indicating the length of the board side.
        pi: A TF tensor for the search probabilities.
        v: A TF tensor for the search values.
        board_feature: A TF tensor with the dimensions of the board.
        training: A TF boolean scalar tensor.
        pi_target: A TF tensor for the target search probabilities.
        v_target: A TF tensor for the target search values.
        loss_pi: A TF tensor for the output of softmax cross entropy on pi.
        loss_v: A TF tensor for the output of mean squared error on v.
        total_loss: A TF tensor to store the addition of pi and v losses.
        train_step: A TF tensor for the train step of the optimizer.
    """

    def __init__(self, game):
        self.side = game.side
        self.pi = None
        self.v = None

        self.board_feature = tf.placeholder(tf.float32,
                                            shape=[None, self.side, self.side])
        self.training = tf.placeholder(tf.bool)

        # Input Layer
        input_layer = tf.reshape(self.board_feature,
                                 [-1, self.side, self.side, 1])

        # Convolutional Block
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=256,
            kernel_size=[3, 3],
            padding="same",
            strides=1)

        batch_norm1 = tf.layers.batch_normalization(
            inputs=conv1,
            training=self.training
        )

        relu1 = tf.nn.relu(batch_norm1)

        resnet_in_out = relu1

        # Residual Tower
        for i in range(20):
            # Residual Block
            conv2 = tf.layers.conv2d(
                inputs=resnet_in_out,
                filters=256,
                kernel_size=[3, 3],
                padding="same",
                strides=1)

            batch_norm2 = tf.layers.batch_normalization(
                inputs=conv2,
                training=self.training
            )

            relu2 = tf.nn.relu(batch_norm2)

            conv3 = tf.layers.conv2d(
                inputs=relu2,
                filters=256,
                kernel_size=[3, 3],
                padding="same",
                strides=1)

            batch_norm3 = tf.layers.batch_normalization(
                inputs=conv3,
                training=self.training
            )

            resnet_skip = tf.add(batch_norm3, resnet_in_out)

            resnet_in_out = tf.nn.relu(resnet_skip)

        # Policy Head
        conv4 = tf.layers.conv2d(
            inputs=resnet_in_out,
            filters=2,
            kernel_size=[1, 1],
            padding="same",
            strides=1)

        batch_norm4 = tf.layers.batch_normalization(
            inputs=conv4,
            training=self.training
        )

        relu4 = tf.nn.relu(batch_norm4)

        relu4_flat = tf.reshape(relu4, [-1, self.side * self.side * 2])

        self.pi = tf.layers.dense(inputs=relu4_flat,
                                  units=(self.side * self.side) + 1)

        # Value Head
        conv5 = tf.layers.conv2d(
            inputs=resnet_in_out,
            filters=1,
            kernel_size=[1, 1],
            padding="same",
            strides=1)

        batch_norm5 = tf.layers.batch_normalization(
            inputs=conv5,
            training=self.training
        )

        relu5 = tf.nn.relu(batch_norm5)

        relu5_flat = tf.reshape(relu5, [-1, 256])

        dense1 = tf.layers.dense(inputs=relu5_flat,
                                 units=1024)

        relu6 = tf.nn.relu(dense1)

        dense2 = tf.layers.dense(inputs=relu6,
                                 units=1)

        self.v = tf.nn.tanh(dense2)

        # Loss Function
        self.pi_target = tf.placeholder(tf.float32,
                                        shape=[None, self.side * self.side + 1])
        self.v_target = tf.placeholder(tf.float32, shape=[None])

        self.loss_pi = tf.losses.softmax_cross_entropy(self.pi_target, self.pi)
        self.loss_v = tf.losses.mean_squared_error(self.v_target,
                                                   tf.reshape(self.v,
                                                              shape=[-1, ]))
        self.total_loss = self.loss_pi + self.loss_v

        # Stochastic gradient descent with momentum and learning rate annealing
        global_step = tf.Variable(0, trainable=False)

        learning_rate = tf.train.exponential_decay(CFG.learning_rate,
                                                   global_step,
                                                   200000,
                                                   0.96,
                                                   staircase=True)

        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                               momentum=CFG.momentum_parameter,
                                               use_nesterov=False)

        train_op = optimizer.minimize(self.total_loss,
                                      global_step=global_step)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            self.train_step = train_op
