import h5py
import googlenet_ext as ge
import cv2
import math
import numpy as np
import os

def create_h5(dataset_path, h5_file_name, pick_gap):
    f = h5py.File(h5_file_name, 'w')
    # video_names is a list of strings containing the 
    # name of a video, e.g. 'video_1', 'video_2'
    # video_path = ['../mydatasets/street.mp4','../mydatasets/walk_3.mp4','../mydatasets/cars.mp4','../mydatasets/walk_1.mp4', '../mydatasets/infor_office.avi']
    # video_path = ['../mydatasets/walk_3.mp4']
    video_path = os.listdir(dataset_path)
    video_names = [p.split('.')[0] for p in video_path]
    video_path = [os.path.join(dataset_path, p) for p in video_path]

    for i in range(len(video_path)):
        print('###### Processing video: {} #######'.format(video_names[i]))
        #Store metadata of video
        cap = cv2.VideoCapture(video_path[i])
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        n_subsample = math.ceil(n_frames/pick_gap)
        features = ge.extract_feature(video_path[i], 'googlenet', pick_gap)

        #create h5 dataset
        f.create_dataset(video_names[i] + '/features', data=features)
        f.create_dataset(video_names[i] + '/gtscore', data=np.zeros([n_subsample], dtype=float))
        f.create_dataset(video_names[i] + '/user_summary', data=np.zeros([15,n_frames]))
        # f.create_dataset(video_names[i] + '/change_points', data=None)
        # f.create_dataset(video_names[i] + '/n_frame_per_seg', data=data_of_name)
        f.create_dataset(video_names[i] + '/n_frames', data=n_frames)
        # f.create_dataset(video_names[i] + '/picks', data=range(0,n_frames, pick_gap))
        f.create_dataset(video_names[i] + '/n_steps', data=n_subsample)
        f.create_dataset(video_names[i] + '/gtsummary', data=np.zeros([n_subsample], dtype=float))
        f.create_dataset(video_names[i] + '/video_name', data=video_names[i])
    f.close()

if __name__== '__main__':
    create_h5('../mydatasets/video', '../mydatasets/vsumm_custom.h5', 10)