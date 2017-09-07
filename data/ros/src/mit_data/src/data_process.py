#!/usr/bin/env python

import rospy
from sensor_msgs.msg import LaserScan, Image
from cv_bridge import CvBridge
import numpy as np
import cv2
import math

WIDTH = 64
HEIGHT = 64



def laser_callback(data):
    img = np.zeros((HEIGHT, WIDTH), np.uint8)
    img[:, :] = 255
    angle = data.angle_min
    angle_increment = data.angle_increment
    for i in range(len(data.ranges)):
        if angle > data.angle_max: break
        x = int(data.ranges[i] * math.cos(angle)) + 24
        y = int(data.ranges[i] * math.sin(angle)) + 32
        if x < 0: x = 0
        if x > 63: x = 63
        if y < 0: y = 0
        if y > 63: y = 63
        img[x, y] = data.intensities[i]
        angle += angle_increment
    cv2.imwrite('/media/hao/hao/dataset/laser_images/' + str(data.header.stamp.secs) + str(data.header.seq) + '.jpg', img)

def image_callback(data):
    bridge = CvBridge()
    img = bridge.imgmsg_to_cv2(data, 'bgr8')
    cv2.imwrite('/media/hao/hao/dataset/images/' + str(data.header.stamp.secs) + str(data.header.seq) + '.jpg', img)



def sub_mit():
    rospy.init_node('sub_mit')

    rospy.Subscriber("base_scan", LaserScan, laser_callback)
    #rospy.Subscriber("camera/rgb/image_raw", Image, image_callback)

    rospy.spin()



if __name__ == '__main__':
    sub_mit()



