__author__ = "Hao Qin"
__email__ = "awww797877@gmail.com"

import numpy as np
from data import dummy
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

BATCH_SIZE = 32

def get_y(poses, index_pose1, index_pose2, num_batch_images, max_transform, max_rotation):
    transform = np.square(
        poses[index_pose1:index_pose1 + num_batch_images, 0] - poses[index_pose2:index_pose2 + num_batch_images,
                                                                   0])
    transform += np.square(
        poses[index_pose1:index_pose1 + num_batch_images, 1] - poses[index_pose2:index_pose2 + num_batch_images,
                                                                   1])
    # print(transform.shape)
    transform = np.sqrt(transform)
    rotation = np.absolute(
        poses[index_pose1:index_pose1 + num_batch_images, 2] - poses[index_pose2:index_pose2 + num_batch_images,
                                                                   2])
    y = 20 * (transform + rotation) / (max_transform + max_rotation) # 1 / (1 + np.exp(- (transform + rotation)))
    y = (1 / (1 + np.exp(y)) - 0.5) * 2
    return y

if __name__ == '__main__':
    poses = dummy.load_poses('/home/hao/others/data/CNN_SLAM/2012-04-06-11-15-29_part1_floor2.gt.laser.poses')

    max_pose = np.amax(poses, axis=0)
    min_pose = np.amin(poses, axis=0)
    max_transform = np.sqrt(np.square(max_pose[0] - min_pose[0]) + np.square(max_pose[1] - min_pose[1]))
    max_rotation = max_pose[2] - min_pose[2]
    print(max_transform, max_rotation)

    z = np.array([])
    for offset1 in range(0, BATCH_SIZE, BATCH_SIZE):
        temp = np.array([])
        for offset2 in range(0, len(poses[0:1216]), BATCH_SIZE):
            t = get_y(poses, offset1, offset2, BATCH_SIZE, max_transform, max_rotation)
            print(t)
            temp = np.append(temp, t)
        z = np.vstack((z, temp)) if z.size else temp

