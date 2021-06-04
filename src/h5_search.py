import argparse

import h5py
import numpy as np
import os
from kts.cpd_auto import cpd_auto
import cv2


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str, required=True)
    # args = parser.parse_args()
    dt = '../datasets/eccv16_dataset_summe_google_pool5.h5'
    h5in = h5py.File(dt, 'r')
    # h5out = h5py.File(args.dataset + '.custom', 'w')
    dic = {}
    for video_name, video_file in h5in.items():
        # n_frames = video_file['n_frames']
        n_frames = video_file['n_frames'][...].astype(int).item()        
        # print(video_name, n_frames)
        if n_frames not in dic:
            dic[n_frames] = [video_name]

    # new_li = sorted(dic.keys())
    # print(dic[new_li[0]])
    for i in dic:
        print(i,dic[i])
    h5in.close()
    other = {}
    dataset_path = '../mydatasets/SUMME_Origin/'
    for d in os.listdir(dataset_path):
        cap = cv2.VideoCapture(os.path.join(dataset_path, d))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if count in dic:
            dic[count].append(d)
        else:
            if count not in other:
                other[count] = d

        # 釋放所有資源
        cap.release()
        cv2.destroyAllWindows()
    
    # h5out.close()
    print('dic:')
    for i in dic:
        print(i, dic[i])
    print('other:')
    for i in other:
        print(i, other[i])
    

if __name__ == '__main__':
    main()