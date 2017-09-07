__author__ = "Hao Qin"
__email__ = "awww797877@gmail.com"
import time
import cv2
import numpy as np
import tensorflow as tf

from network.network import Network
from network import blocks
from data import dummy

from network.network import Network
from network import blocks
from data import dummy

BATCH_SIZE = 1

def image_process(images, laser_images):
    processed_images = np.array([])
    num = len(images) if len(images) <= len(laser_images) else len(laser_images)
    for i in range(num):
        img = cv2.resize(images[i], dsize=(64, 48))
        processed_image = np.concatenate((img, laser_images[i]), axis=2)[np.newaxis, :]
        processed_images = np.vstack((processed_images, processed_image)) if processed_images.size else processed_image
    return processed_images

if __name__ == '__main__':
    beginTime = time.time()
    images = dummy.load_images('/home/hao/others/data/CNN_SLAM/images')
    laser_images = dummy.load_laser_images('/home/hao/others/data/CNN_SLAM/laser_images')
    size = len(images) if len(images) <= len(laser_images) else len(laser_images)
    images = images[0:size]
    laser_images = laser_images[0:size]
    processed_images = image_process(images, laser_images)

    num_images = len(processed_images)
    shape = processed_images[0].shape

    tf_images = tf.placeholder(shape=[None, shape[0], shape[1], shape[2] * 2], dtype=tf.float32, name='images')
    tf_batch_size = tf.placeholder(dtype=tf.float32, name='batch_size')
    network = Network()
    output = network.inference(tf_images, tf_batch_size, 'images')

    sess = tf.InteractiveSession()
    with sess.as_default():
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, '/home/hao/others/CNN_SLAM/check_points/snap.ckpt-0')
        for offset1 in range(0, num_images, BATCH_SIZE):
            end1 = offset1 + BATCH_SIZE
            for offset2 in range(0, num_images, BATCH_SIZE):
                end2 = offset2 + BATCH_SIZE
                if num_images - offset1 < BATCH_SIZE:
                    tf_batch_images1 = processed_images[offset1:end1]
                    tf_batch_images2 = processed_images[offset2:offset2 + num_images - offset1]
                elif (num_images - offset2) < BATCH_SIZE:
                    tf_batch_images1 = processed_images[offset1:offset1 + num_images - offset2]
                    tf_batch_images2 = processed_images[offset2:end2]
                else:
                    tf_batch_images1 = processed_images[offset1:end1]
                    tf_batch_images2 = processed_images[offset2:end2]
                tf_batch_images = np.concatenate((tf_batch_images1, tf_batch_images2), axis=3)
                feed_dict = {
                    tf_images: tf_batch_images,
                    tf_batch_size: len(tf_batch_images1),
                }
                batch_out = sess.run([output], feed_dict=feed_dict)
                #batch_loss = sess.run([loss], feed_dict=feed_dict)
                print("Now compare {0} and {1}, the batch_out is: {2}".format(offset1, offset2, batch_out))