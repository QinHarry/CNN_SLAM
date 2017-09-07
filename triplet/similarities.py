import numpy as np
import os
import glob
import time
import cv2
import torch
import argparse
from tqdm import tqdm
from torch.autograd import Variable

from t_net import TNet

parser = argparse.ArgumentParser(description='Computing the similarities of each image')

parser.add_argument('--input_path', type=str, \
                    default='/media/hao/hao/dataset/images/', \
                    help='The input path of images')
parser.add_argument('--output_path', type=str, \
                    default='/media/hao/hao/dataset/similarities.txt', help='The output path of similarities')
parser.add_argument('--model', type=str, \
                    default='/home/hao/Documents/CNN_SLAM/triplet/models/checkpoint_0.pth', help='The model for prediction')

args = parser.parse_args()

if __name__ == '__main__':
    input_path = args.input_path
    output_path = args.output_path

    t = time.time()

    t_net = TNet()
    t_net.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage)['state_dict'])
    t_net.eval()

    poses = np.array([
        [29.3272,129.709,0.381379],
        [29.3275,129.71,0.381226],
        [17.2895,142.901,2.03573],
        [14.2357,145.707,2.73077],
        [5.85108,145.835,-2.21034]
    ])

    img_list = [
        '13337361334.jpg',
        '133373613333.jpg',
        '13337363977923.jpg',
        '13337364068193.jpg',
        '13337364298883.jpg'
    ]

    laser_img_list = [
        '13337361335408.jpg',
        '13337361335427.jpg',
        '133373639710684.jpg',
        '133373640610864.jpg',
        '133373642911323.jpg'
    ]

    inputs = np.array([])
    for i in range(len(img_list)):
        input = cv2.imread(os.path.join(input_path, img_list[i]), 0)
        input = cv2.resize(input, dsize=(32, 32), interpolation=cv2.INTER_LINEAR)[np.newaxis, :]
        inputs = np.vstack((inputs, input)) if inputs.size else input
    inputs = inputs.reshape((5, 1, 32, 32))

    t = time.time()
    for i in range(len(img_list)):
        for j in range(len(img_list)):
            if i == j: continue
            dist = np.sqrt(np.sum(np.square(poses[i] - poses[j])))
            input_i = torch.from_numpy(inputs[i:i+1, :, :, :])
            input_i = Variable(input_i).float()
            prediction1 = t_net.forward(input_i)
            prediction1 = prediction1.data.cpu().numpy()
            input_j = torch.from_numpy(inputs[j:j+1, :, :, :])
            input_j = Variable(input_j).float()
            prediction2 = t_net.forward(input_j)
            prediction2 = prediction2.data.cpu().numpy()
            dist_d = np.sqrt(np.sum(np.square(prediction1 - prediction2)))
            print('Compare {0} and {1}, the pose distance is: {2}, the descriptors is: {3}'.format(i, j, dist, dist_d))
    t1 = time.time()
    print(t1 - t)
    # input_list = os.path.join(input_path, '*.jpg')
    # input_list = glob.glob(input_list)
    #
    # inputs = np.array([])
    # for i in tqdm(range(len(input_list))):
    #     if i > 0 and i % 450 == 0:
    #         input = cv2.imread(input_list[i], 0)
    #         input = cv2.resize(input, dsize=(32, 32), interpolation=cv2.INTER_LINEAR)[np.newaxis, :]
    #         inputs = np.vstack((inputs, input)) if inputs.size else input
    # print(inputs.shape)
