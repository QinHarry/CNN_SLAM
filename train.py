__author__ = "Hao Qin"
__email__ = "awww797877@gmail.com"
import time
import cv2
import numpy as np
import tensorflow as tf
import random
import scipy.ndimage

from network.network import Network
from network import blocks
from data import dummy

EPOCHS = 2
BATCH_SIZE = 32
RATE = 0.000000005
BETA = 0.0005

def image_process(images, laser_images):
    processed_images = np.array([])
    num = len(images) if len(images) <= len(laser_images) else len(laser_images)
    for i in range(num):
        img = cv2.resize(images[i], dsize=(64, 48))
        processed_image = np.concatenate((img, laser_images[i]), axis=2)[np.newaxis, :]
        processed_images = np.vstack((processed_images, processed_image)) if processed_images.size else processed_image
    return processed_images

def images_expand(images):
    if len(images) == 0: return images
    new_images = images
    k = 0
    for i in range(3):
        print('#################################')
        print('Now process {0}'.format(i))
        print('#################################')
        for image in images:
            k += 1
            new_image = transform_image(image,10,5,2,brightness=1)[np.newaxis, :]
            new_images = np.vstack((new_images, new_image))
            print('*** The images number: {0} ***'.format(k))
    return new_images

def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def transform_image(img,ang_range,shear_range,trans_range,brightness=0):
    '''
    This function transforms images to generate new images.
    The function takes in following arguments,
    1- Image
    2- ang_range: Range of angles for rotation
    3- shear_range: Range of values to apply affine transform to
    4- trans_range: Range of values to apply translations over.

    A Random uniform distribution is used to generate different parameters for transformation

    '''
    # Rotation

    ang_rot = np.random.uniform(ang_range)-ang_range/2
    rows,cols,ch = img.shape
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)

    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

    # Shear
    pts1 = np.float32([[trans_range,trans_range],[ang_range,trans_range],[trans_range,ang_range]])

    pt1 = trans_range+shear_range*np.random.uniform()-shear_range/2
    pt2 = ang_range+shear_range*np.random.uniform()-shear_range/2

    # Brightness


    pts2 = np.float32([[pt1,trans_range],[pt2,pt1],[trans_range,pt2]])

    shear_M = cv2.getAffineTransform(pts1,pts2)

    img = cv2.warpAffine(img,Rot_M,(cols,rows))
    img = cv2.warpAffine(img,Trans_M,(cols,rows))
    img = cv2.warpAffine(img,shear_M,(cols,rows))

    if brightness == 1:
      img = augment_brightness_camera_images(img)

    return img

def transform_processed_image(input,ang_range):
    # Rotation
    ang_rot = np.random.uniform(ang_range) - ang_range / 2
    new_image = scipy.ndimage.rotate(input, ang_rot, reshape = False)

    # TODO Plus transform and shear
    # # Translation
    # tr_x = trans_range * np.random.uniform() - trans_range / 2
    # tr_y = trans_range * np.random.uniform() - trans_range / 2

    # shear
    return new_image



def get_y(poses, index_pose1, index_pose2, max_transform, max_rotation):
    transform = np.square(
        poses[index_pose1, 0] - poses[index_pose2, 0])
    transform += np.square(
        poses[index_pose1, 1] - poses[index_pose2, 1])
    # print(transform.shape)
    transform = np.sqrt(transform)
    rotation = np.absolute(
        poses[index_pose1, 2] - poses[index_pose2, 2])
    y = (transform + rotation) / (max_transform + max_rotation)  # 1 / (1 + np.exp(- (transform + rotation)))
    y = 4 * (1 / (1 + np.exp(-y)) - 0.5) * 2
    # for i in range(len(y)):
    #     if y[i] < 0.1: y[i] = 0.
    #     else: y[i] = -1.
    y = y
    return y


if __name__ == '__main__':
    beginTime = time.time()
    images = dummy.load_images('/home/hao/others/data/CNN_SLAM/images')
    laser_images = dummy.load_laser_images('/home/hao/others/data/CNN_SLAM/laser_images')
    size = len(images) if len(images) <= len(laser_images) else len(laser_images)
    images = images[0:size]
    laser_images = laser_images[0:size]
    # expanded_images = images_expand(images)
    # for i in range(2):
    #     laser_images = np.vstack((laser_images, laser_images))
    processed_images = image_process(images, laser_images)
    poses = dummy.load_poses('/home/hao/others/data/CNN_SLAM/2012-04-06-11-15-29_part1_floor2.gt.laser.poses')
    # for i in range(2):
    #     poses = np.vstack((poses, poses))


    num_images = len(processed_images)
    num_poses = len(poses)
    shape = processed_images[0].shape

    max_pose = np.amax(poses, axis=0)
    min_pose = np.amin(poses, axis=0)
    max_transform = np.sqrt(np.square(max_pose[0] - min_pose[0]) + np.square(max_pose[1] - min_pose[1]))
    max_rotation = max_pose[2] - min_pose[2]

    tf_images = tf.placeholder(shape=[None, shape[0], shape[1], shape[2] * 2], dtype=tf.float32, name='images')
    tf_batch_size = tf.placeholder(dtype=tf.int32, name='batch_size')
    tf_y = tf.placeholder(tf.float32, (None), name='y')
    network = Network()
    output = network.inference(tf_images, tf_batch_size, 'images')

    #one_hot_y = tf.one_hot(tf_y, 2)

    l2 = blocks.l2_regulariser(decay=BETA)
    loss = network.loss(output, tf_batch_size, tf_y)
    loss = loss
    # loss = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=output)
    # loss = tf.reduce_mean(loss)

    optimizer = tf.train.MomentumOptimizer(learning_rate=RATE, momentum=0.9)
    training_operation = optimizer.minimize(loss)

    sess = tf.InteractiveSession()
    with sess.as_default():
        saver = tf.train.Saver()
        train_writer = tf.summary.FileWriter('tf_logs' + '/train',
                                             sess.graph)
        sess.run(tf.global_variables_initializer())
        print("Training...")
        print()
        for k in range(EPOCHS):

            index1 = random.sample(range(num_images), BATCH_SIZE)
            index2 = random.sample(range(num_images), BATCH_SIZE)
            tf_batch_images = np.concatenate((processed_images[index1], processed_images[index2]), axis=3)
            y = get_y(poses, index1, index2, max_transform, max_rotation) # [np.newaxis, :].T
            for i in range(len(y)):
                if y[i] < 0.2: y[i] = -1
                else: y[i] = 1
            matched_id = np.array(np.where(y <= 0.2))
            if matched_id.shape[1] == 0: continue
            for i in range((BATCH_SIZE - matched_id.shape[1]) / matched_id.shape[1] - 1):
                for id in matched_id[0]:
                    new_image = transform_processed_image(tf_batch_images[id], 40)[np.newaxis, :]
                    tf_batch_images = np.vstack((tf_batch_images, new_image))
                    y = np.append(y, 1)
            feed_dict = {
                tf_images: tf_batch_images,
                tf_batch_size: len(tf_batch_images),
                tf_y: y,
            }
            b_output = sess.run([output], feed_dict={tf_images: tf_batch_images, tf_batch_size: BATCH_SIZE})
            print(b_output)
            # sess.run([training_operation], feed_dict=feed_dict)
            # if k % 100 == 0:
            #     batch_loss = sess.run([loss], feed_dict=feed_dict)
            #     print('At {0}, the loss is: {1}'.format(k, batch_loss))
            #     saver.save(sess, '/home/hao/others/CNN_SLAM/check_points/snap.ckpt', global_step=0)









