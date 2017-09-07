__author__ = "Hao Qin"
__email__ = "awww797877@gmail.com"

import os
import glob
import numpy as np
import cv2

rate = 16328 / 12361.

def load_images(path):
    imgs = np.array([])
    imgs_list = os.path.join(path, '*.jpg')
    imgs_list = np.array(glob.glob(imgs_list))
    imgs_id = np.array([])
    for img_list in imgs_list:
        imgs_id = np.append(imgs_id, int(os.path.basename(img_list).split('.')[0]))
    sorted_id = imgs_id.argsort()
    for i in range(64):
        i_rate = int(i * rate)
        temp = cv2.imread(imgs_list[sorted_id[i_rate]])
        cv2.normalize(temp, temp, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        temp = temp[np.newaxis, :]
        imgs = np.vstack((imgs, temp)) if imgs.size else temp
        print('Now read the image: {0}'.format(str(imgs_id[sorted_id[i_rate]])))

    return imgs

def load_laser_images(path):
    imgs = np.array([])
    imgs_list = os.path.join(path, '*.jpg')
    imgs_list = np.array(glob.glob(imgs_list))
    imgs_id = np.array([])
    for img_list in imgs_list:
        imgs_id = np.append(imgs_id, int(os.path.basename(img_list).split('.')[0]))
    sorted_id = imgs_id.argsort()
    print(imgs_id.shape)
    for i in range(64): #
        temp = cv2.imread(imgs_list[sorted_id[i]])
        cv2.normalize(temp, temp, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        temp = temp[np.newaxis, :]
        imgs = np.vstack((imgs, temp)) if imgs.size else temp
        print('Now read the laser image: {0}'.format(str(imgs_id[sorted_id[i]])))
    return imgs

def load_poses(path):
    poses = np.array([])
    with open(path) as f:
        content = f.readlines()
    for i in range(64):
        temp = np.array(content[i].split(','))[np.newaxis, :]
        poses = np.vstack((poses, temp)) if poses.size else temp
    poses = poses.astype(np.float)
    return poses[:, 1:]

if __name__ == '__main__':
    #load_images('/media/hao/hao/dataset/images')
    #load_laser_images('/media/hao/hao/dataset/laser_images')
    poses = load_poses('/home/hao/others/data/CNN_SLAM/2012-04-06-11-15-29_part1_floor2.gt.laser.poses')
    print(np.amax(poses, axis=0))
    print(np.amin(poses, axis=0))