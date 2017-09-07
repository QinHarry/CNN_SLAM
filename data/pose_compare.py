#!/usr/bin/env python

__author__ = "Hao Qin"
__email__ = "awww797877@gmail.com"

import os
import glob
import numpy as np
import re
import matplotlib.pyplot as plt
import math

def submaps_pose_combine(path, output_path):
    submap_files = os.path.join(path, '*.txt')
    submap_files = np.array(glob.glob(submap_files))
    with open(output_path, 'w') as fw:
        for submap in submap_files:
            with open(submap, 'r') as f:
                content = f.readlines()
                one_line = ''
                for i in range(2):
                    one_line = one_line + content[i].split()[1] + '\t'
                fw.write(one_line + '\n')

def compute_pose_from_cartographer(path_submaps, path_contraints, path_node_pose):
    submaps_x = []
    submaps_y = []
    node_x = [0]*2832
    node_y = [0]*2832
    with open(path_submaps, 'r') as f:
        submaps = f.readlines()
    if len(submaps) <= 0: return
    for submap in submaps:
        submaps_x.append(float(submap.split()[0]))
        submaps_y.append(float(submap.split()[1]))
    with open(path_contraints, 'r') as f:
        contraints = f.readlines()
        pattern_index = re.compile(r"\,(.*?)\)", re.I | re.X)
        pattern_trans = re.compile(r"t:.\[(.*?)\]", re.I | re.X)
        for contraint in contraints:
            index = pattern_index.findall(contraint)
            submap_index = int(index[0])
            node_index = int(index[1])
            pose = pattern_trans.findall(contraint)
            node_x[node_index] = submaps_x[submap_index] + float(pose[0].split(',')[0])
            node_y[node_index] = submaps_y[submap_index] + float(pose[0].split(',')[1])
    with open(path_node_pose, 'w') as f:
        for i in range(2832):
            f.writelines(str(node_x[i]) + ' ' + str(node_y[i]) + '\n')

def compute_groud_truth(poses_path, groud_truth_path):
    with open(poses_path, 'r') as f:
        poses = f.readlines()
    base_x = float(poses[74].split(',')[1])
    base_y = float(poses[74].split(',')[2])
    base_angle = float(poses[74].split(',')[3])
    groud_truth = []
    for i in range(74, len(poses)):
        x = float(poses[i].split(',')[1]) - base_x
        y = float(poses[i].split(',')[2]) - base_y
        angle = float(poses[i].split(',')[3]) - base_angle
        groud_truth.append(str(poses[i].split(',')[0]) + '\t' + str(x) + '\t' + str(y) + '\t' + str(angle) + '\n')
    with open(groud_truth_path, 'w') as f:
        for i in groud_truth:
            f.writelines(i)

def draw_compare(path_node, path_ground_truth):
    with open(path_node) as f:
        node_poses_str = f.readlines()
    node_pose_x = []
    node_pose_y = []
    for i in range(len(node_poses_str)):
        node_pose_x.append((float(node_poses_str[i].split()[1]) - 3.7))
        node_pose_y.append(float(node_poses_str[i].split()[2])) #  + (float(node_poses_str[i].split()[1]) - 7.3)*0.5
    with open(path_ground_truth) as f:
        ground_truth_str = f.readlines()
    ground_truth_x = []
    ground_truth_y = []
    for i in ground_truth_str:
        ground_truth_x.append(float(i.split()[1]))
        ground_truth_y.append(float(i.split()[2]))
    plt.figure()
    #node_pose_y_linspace = np.linspace(0, len(ground_truth_y), len(node_pose_y))
    #ground_truth_y_linspace = np.linspace(0, len(ground_truth_y), len(ground_truth_y))
    plt.plot(node_pose_x, node_pose_y, 'r-x', label='node_pose')
    plt.plot(ground_truth_x, ground_truth_y, 'g-^', label='ground_truth')
    plt.legend()
    plt.xlabel('pose_x')
    plt.ylabel('pose_y')
    plt.title('Pose compare between prediction and ground truth')
    plt.show()

def draw_compare_angle(path_node, path_ground_truth):
    with open(path_node) as f:
        node_poses_str = f.readlines()
    node_pose_angle = []
    for i in range(len(node_poses_str)):
        node_pose_angle.append((float(node_poses_str[i].split()[3])))
    with open(path_ground_truth) as f:
        ground_truth_str = f.readlines()
    ground_truth_angle = []
    for i in ground_truth_str:
        ground_truth_angle.append(float(i.split()[3]))
    plt.figure()
    node_pose_linspace = np.linspace(0, len(ground_truth_angle), len(node_pose_angle))
    ground_truth_linspace = np.linspace(0, len(ground_truth_angle), len(ground_truth_angle))
    plt.plot(node_pose_linspace, node_pose_angle, 'r-x', label='node_pose')
    plt.plot(ground_truth_linspace, ground_truth_angle, 'g-^', label='ground_truth')
    plt.legend()
    plt.xlabel('pose_x')
    plt.ylabel('pose_y')
    plt.title('Pose compare between prediction and ground truth')
    plt.show()

def pose_distance(path_ground_truth):
    with open(path_ground_truth) as f:
        ground_truth_str = f.readlines()
    pose_sample = []
    base_timestamp = ground_truth_str[0].split(',')[0]
    base_x = ground_truth_str[0].split(',')[1]
    base_y = ground_truth_str[0].split(',')[2]
    for i in ground_truth_str:
        content = i.split(',')
        timestamp = content[0]
        x = content[1]
        y = content[2]









if __name__ == '__main__':
    # submaps_pose_combine('/media/hao/hao/dataset/submaps', '/media/hao/hao/dataset/submaps.txt')
    # compute_pose_from_cartographer('/media/hao/hao/dataset/submaps.txt', '/media/hao/hao/dataset/constraints.txt', '/media/hao/hao/dataset/node_pose.txt')
    # compute_groud_truth('/media/hao/hao/dataset/2012-04-06-11-15-29_part1_floor2.gt.laser.poses', '/media/hao/hao/dataset/ground_truth.txt')
    draw_compare_angle('/media/hao/hao/dataset/ros_pose.txt', '/media/hao/hao/dataset/ground_truth.txt')
    #pose_distance('/media/hao/hao/dataset/ground_truth.txt')

