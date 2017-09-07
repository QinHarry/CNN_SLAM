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

TFEAT_PATCH_SIZE = 32
TFEAT_DESC_SIZE = 128
TFEAT_BATCH_SIZE = 1000
MEAN = 0.48544601108437
STD = 0.18649942105166
MODEL_FNAME = '/home/hao/others/triplet/logs/run-optim_sgd-n1280000-lr0.1-wd0.0001-m2.0-S0-tanh/checkpoint_8.pth'


parser = argparse.ArgumentParser(description='Extract descriptors from hpatch file')

parser.add_argument('--input_path', type=str, \
                    default='/home/hao/others/triplet/hpatches-benchmark/data/hpatches-release/', \
                    help='The input path of images')
parser.add_argument('--output_path', type=str, \
                    default='/home/hao/others/triplet/hpatches-benchmark/data/descriptors/my_tfeat/', help='The output path of descriptors')
args = parser.parse_args()



def preprocess_patch(patch):
    out = cv2.resize(patch, (TFEAT_PATCH_SIZE, TFEAT_PATCH_SIZE)).astype(np.float32) / 255;
    out = (out - MEAN) / STD
    return out.reshape(1, TFEAT_PATCH_SIZE, TFEAT_PATCH_SIZE)

def extract_tfeats(net,patches):
    num,channels,h,w = patches.shape
    patches_t = torch.from_numpy(patches)
    patches_t = Variable(patches_t).float()

    descriptors = []
    for i in tqdm(range(num)):
        prediction = net.forward(patches_t[i:i+1,:,:,:])
        prediction = prediction.data.cpu().numpy()
        descriptors.append(prediction)
    out = np.concatenate(descriptors)
    return out.reshape(num, TFEAT_DESC_SIZE)



if __name__ == '__main__':

    input_path = args.input_path
    output_path = args.output_path
    t = time.time()

    t_net = TNet()
    t_net.load_state_dict(torch.load(MODEL_FNAME)['state_dict'])
    t_net.eval()

    input_list = os.path.join(input_path, '*')
    input_list = glob.glob(input_list)
    input_list = input_list
    for seq in tqdm(input_list):
        input_seq = os.path.join(seq, '*.png')
        input_seq = glob.glob(input_seq)
        out_seq = os.path.join(output_path, os.path.basename(seq))
        if not os.path.exists(out_seq):
            os.makedirs(out_seq)
        for img_path in tqdm(input_seq):
            out_descrs_path = os.path.join(out_seq, os.path.basename(img_path))
            out_descrs_path = out_descrs_path.split('.')[0] + '.csv'
            img = cv2.imread(img_path, 0)
            h, w = img.shape
            n_patches = h / w
            patches = np.zeros((n_patches, 1, TFEAT_PATCH_SIZE, TFEAT_PATCH_SIZE))
            for i in range(n_patches):
                patches[i, :, :, :] = preprocess_patch(img[i * (w): (i + 1) * (w), 0:w])
            out_descs = extract_tfeats(t_net, patches)
            np.savetxt(out_descrs_path, out_descs, delimiter=',', fmt='%10.7f')
