__author__ = "Hao Qin"
__email__ = "awww797877@gmail.com"

import blocks
import tensorflow as tf

class Network:

    def __init__(self):
        pass

    def inference(self, input, batch_size, name):
        block = blocks.conv2d_batch_norm_relu(input, out_channels=96, kernel_size=(5, 5), strides=[1, 1, 1, 1], padding = 'SAME', batch_size=batch_size, name=name + 'conv1')
        block = blocks.maxpool(block, kernel_size=(2, 2), strides=[1, 2, 2, 1], padding = 'SAME', name=name + 'pool1')
        block = blocks.conv2d_batch_norm_relu(block, out_channels=96, kernel_size=(3, 3), strides=[1, 1, 1, 1], padding='SAME', batch_size=batch_size, name=name + 'conv2')
        block = blocks.maxpool(block, kernel_size=(2, 2), strides=[1, 2, 2, 1], padding='SAME', name=name + 'pool2')
        block = blocks.conv2d_batch_norm_relu(block, out_channels=192, kernel_size=(3, 3), strides=[1, 1, 1, 1], padding='SAME', batch_size=batch_size,
                              name=name + 'conv3')
        block = blocks.conv2d_batch_norm_relu(block, out_channels=192, kernel_size=(3, 3), strides=[1, 1, 1, 1],
                                              padding='SAME', batch_size=batch_size,
                                              name=name + 'conv4')

        block1 = blocks.average_pool(block, kernel_size=(2, 2), strides=[1, 2, 2, 1], padding='SAME', name=name + 'pool3')
        block2 = tf.image.crop_to_bounding_box(block, 3, 4, 6, 8)

        block = tf.concat([block1, block2], 3)

        block = blocks.flatten(block)

        shape = block.get_shape().as_list()
        block = blocks.linear_relu(block, num_hiddens=shape[1]/2, name='fc1')
        block = blocks.linear(block, num_hiddens=1, name='fc2')




        out = block



        return out

    def loss(self, output, batch_size, y):
        #loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output, name='loss')

        loss = 1 + y * output
        zeros = tf.zeros(shape=[batch_size])
        loss = tf.maximum(loss, zeros)
        loss = tf.reduce_mean(loss)
        #loss = output
        # zeros = tf.zeros(shape=[batch_size])
        # loss = tf.maximum(loss, zeros)
        # def if_true():
        #     return tf.zeros(shape=[1])
        # def if_false():
        #     return loss
        # loss = tf.cond(loss < 0.0, if_true, if_false)

        return loss

    def loss_test(self, descriptor, batch_size):
        loss = tf.sqrt(tf.reduce_sum(tf.square(descriptor))) / batch_size

        return loss