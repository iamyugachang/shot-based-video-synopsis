import os
####################################
## This file contains funtions:
##     1. main(None)-> None 
##     2. evaluate(model, video data, other parameter)-> prediction result
##     whole file:
##         Input: 
##             1. the h5 file (which contains many videos info)
##             2. original video path
##         Ouptut: 
##             1. summary video
######################################

import logging
from pathlib import Path

import numpy as np
import torch

from helpers import init_helper, data_helper, vsumm_helper, bbox_helper
from modules.model_zoo import get_model
import cv2
import os
import h5py
from collections import Counter 
from tqdm import tqdm
logger = logging.getLogger()


def evaluate(model, val_loader, nms_thresh, device):
    model.eval()
    pred_li = {}
    with torch.no_grad():
        for val in val_loader:
            # Read h5 file and collect all parameters
            test_key, seq, _, cps, n_frames, nfps, picks,_ = val  
            seq_len = len(seq) #length of features extracted from googlenet (shape = (n_frames/subsample_amount, 1024))
            seq_torch = torch.from_numpy(seq).unsqueeze(0).to(device)   

            # Predict confidence of classes and bbox position         
            pred_cls, pred_bboxes = model.predict(seq_torch)
            pred_bboxes = np.clip(pred_bboxes, 0, seq_len).round().astype(np.int32) #transform all elements in bboxes to range in (0, len(seq)), 
            pred_cls, pred_bboxes = bbox_helper.nms(pred_cls, pred_bboxes, nms_thresh) #filter score larger than nms_thresh in cls & bboxes
            pred_summ = vsumm_helper.bbox2summary(
                seq_len, pred_cls, pred_bboxes, cps, n_frames, nfps, picks)
            
            # Store pred_summ ex: pred_li = {'video_1', []}
            if test_key.split('/')[-1] not in pred_li:
                pred_li[test_key.split('/')[-1]] = pred_summ

    return pred_li


def main(version):
    # initial parameters
    # args.input_data = '../mydatasets/vsumm_custom_after_kts.h5'
    # args.output_folder = '../output/'
    # args.model_path = '../models/pretrain_ab_basic/checkpoint/summe.yml.4.pt' #could be other pt model
    # args.input_video = '../mydatasets/video/'

    # Load model argument
    args = init_helper.get_arguments()
    init_helper.init_logger(args.model_dir, args.log_file) #arg.model_dir = ../models/pretrain_ab_basic
    init_helper.set_random_seed(args.seed)
    logger.info(vars(args))
    model = get_model(args.model, **vars(args)) 
    model = model.eval().to(args.device)
    
    # Load h5 data
    h5in = h5py.File(args.input_data, 'r')
    ckpt_path = args.model_path
    state_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    val_set = [os.path.join(args.input_data, video_name) for video_name, _ in h5in.items()]
    val_set = data_helper.VideoDataset(val_set)
    val_loader = data_helper.DataLoader(val_set, shuffle=False)
    pred_li = evaluate(model, val_loader, args.nms_thresh, args.device)


    # Map original video with h5 data
    video_map = {} 
    for f_name in os.listdir(args.input_video):
        tmp = f_name.split('.')[0]
        video_map[tmp] = f_name
    
    # Generate summary video according to the original videos and predicted summary(frame-level score)
    for file_name in pred_li:
        pred_summ = pred_li[file_name]
        cap = cv2.VideoCapture(os.path.join(args.input_video, video_map[file_name]))
        # cap = cv2.VideoCapture('../mydatasets/myvideo/walk_3.mp4')
        width  = int(cap.get(3))
        height = int(cap.get(4))
        fps = int(cap.get(5))
        size = (width, height)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')    #for avi extension
        output_name =  file_name+'_SUMMARY_OUTPUT'+'.avi'
        # fourcc = cv2.VideoWriter_fourcc(*'MP4V')  #for mp4 extension
        # output_name =  file_name+'_SUMMARY_OUTPUT'+'.mp4'
        output_path = os.path.join(args.output_folder, output_name)
        out = cv2.VideoWriter(output_path, fourcc, fps, size)
        count = 0
        print('Generating summary videos', output_path, 'from', os.path.join(args.input_video, video_map[file_name]))
        for count in tqdm(range(len(pred_summ))):
        # while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                # 寫入影格
                if pred_summ[count]:
                    out.write(frame)
                # count+=1
                    # cv2.imshow('frame',frame)
                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     break
            else:
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # create h5 file (add feature)
    # create shot file (add change_points, n_frame_per_segment)
    # create video file (send h5 file with all data into evaluate)
    main(version='folder') #version = split or folder
    pass