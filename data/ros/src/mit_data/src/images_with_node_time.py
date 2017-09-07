#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

times = []

def image_callback(data):
    time = int(str(data.header.stamp.secs) + str(data.header.stamp.nsecs))
    # if abs(time - int(times[0])) < 50000000:
    #     print(time, times[0])
    if abs(time - int(times[0])) < 100000000:
        bridge = CvBridge()
        img = bridge.imgmsg_to_cv2(data, 'bgr8')
        cv2.imwrite('/media/hao/hao/dataset/images_with_node/' + str(data.header.stamp.secs) + str(data.header.stamp.nsecs) + '.jpg', img)
        times.pop(0)

def sub_mit_images():
    global times
    with open('/media/hao/hao/dataset/node_time.txt') as f:
        times = f.readlines()
    rospy.init_node('sub_mit_images')
    rospy.Subscriber("camera/rgb/image_raw", Image, image_callback)

    rospy.spin()

if __name__ == '__main__':
    sub_mit_images()