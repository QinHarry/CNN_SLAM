#!/usr/bin/env python

import rospy
import tf

if __name__ == '__main__':
    rospy.init_node('pose_saver')
    listener = tf.TransformListener()

    listener.waitForTransform('/map', '/base_footprint', rospy.Time(0), rospy.Duration(1.0))
    rate = rospy.Rate(20.0)


    with open('/media/hao/hao/dataset/ros_pose.txt', 'w') as f:
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            listener.waitForTransform('/map', '/base_footprint', now, rospy.Duration(10.0))
            (trans, rot) = listener.lookupTransform('/map', '/base_footprint', now)

            euler = tf.transformations.euler_from_quaternion(rot)

            f.writelines(str(now) + '\t' + str(trans[0]) + '\t' + str(trans[1]) + '\t' + str(euler[2]) + '\n')
            rate.sleep()
