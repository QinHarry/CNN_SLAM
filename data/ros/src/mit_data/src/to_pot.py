#!/usr/bin/env python

import rospy, math, random
import numpy as np
from sensor_msgs.msg import LaserScan
from laser_geometry import LaserProjection
import sensor_msgs.point_cloud2 as pc2

laser_projector = LaserProjection()

def listener(scan_topic="/base_scan"):
    rospy.init_node('listener')
    rospy.Subscriber(scan_topic, LaserScan, on_scan)
    rospy.spin()

def on_scan(scan):
    rospy.loginfo("Got scan, projecting")
    cloud = laser_projector.projectLaser(scan)
    gen = pc2.read_points(cloud, skip_nans=True, field_names=("x", "y", "z"))
    for p in gen:
	#print " x : %f  y: %f  z: %f" %(p[0],p[1],p[2])
	print(p.size())
    rospy.loginfo("Printed cloud")

if __name__ == '__main__':
    listener()
