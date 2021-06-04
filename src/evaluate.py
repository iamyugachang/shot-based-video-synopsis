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
logger = logging.getLogger()


def evaluate(model, val_loader, nms_thresh, device):
    model.eval()
    pred_li = {}
    # stats = data_helper.AverageMeter('fscore', 'diversity')
    #print("length of val_loader", len(val_loader))
    with torch.no_grad():
        # for test_key, seq, _, cps, n_frames, nfps, picks, user_summary in val_loader: #read a video in h5 format
        for val in val_loader:
            test_key, seq, _, cps, n_frames, nfps, picks,_ = val
            # test_key      -> path of the video                    -> ex: 
            # seq           -> feature (length = n_frames/15)       -> ex: length = 3284/15 = 219
            # n_frames      -> total number of frames in the video  -> ex: 3284
            # picks         ->                                      -> ex: [0,15,30,45,60,...]              
            # user_summary  -> 
            # nfps          -> number of frames per segment         -> ex: [75, 90, 45,...]                 **how to decide segment?
            # cps           -> change points(frame positions)       -> ex: [(0,74), (75,164), (165,209)]
            
            seq_len = len(seq) #seq.shape = (n_frames/15, GoogLeNet output size) ex: (219, 1024)
            seq_torch = torch.from_numpy(seq).unsqueeze(0).to(device)            #-> (1,219,1024)
            pred_cls, pred_bboxes = model.predict(seq_torch)
            # pred_cls      -> [    score1    ,     score2   ,...]  -> [ 0.4  ,   0.6  , ...]
            # pred_bboxs    -> [(pos11, pos12),(pos21, pos22),...]  -> [(0, 2), (-2, 3), ...] (length of pred_bboxs = len(seq)*4)
            pred_bboxes = np.clip(pred_bboxes, 0, seq_len).round().astype(np.int32) #transform all elements in bboxes to range in (0, len(seq)), 
                                                                                    #cuz frames are all in the range
            # pred_cls      -> [    score1    ,     score2   ,...]  -> [ 0.4  ,  0.6  , ...]
            # pred_bboxs    -> [(pos11, pos12),(pos21, pos22),...]  -> [(0, 2), (0, 3), ...]
            # print('frames:', n_frames)
            # print('p1:', pred_bboxes.shape, pred_cls.shape)
            pred_cls, pred_bboxes = bbox_helper.nms(pred_cls, pred_bboxes, nms_thresh) #filter score larger than nms_thresh in cls&bboxes
            # print('p2:', pred_bboxes.shape, pred_cls.shape)
            # print("pred_cls:", pred_cls)
            # print("pred_bboxes", pred_bboxes)
            # break
            pred_summ = vsumm_helper.bbox2summary(
                seq_len, pred_cls, pred_bboxes, cps, n_frames, nfps, picks)
            
            #Storing pred_summ ex: pred_li = {'video_1', []}
            if test_key.split('/')[-1] not in pred_li:
                pred_li[test_key.split('/')[-1]] = pred_summ

            # eval_metric = 'avg' if 'tvsum' in test_key else 'max'
            # fscore = vsumm_helper.get_summ_f1score(
            #     pred_summ, user_summary, eval_metric)

            # pred_summ = vsumm_helper.downsample_summ(pred_summ)
            # diversity = vsumm_helper.get_summ_diversity(pred_summ, seq)
            # stats.update(fscore=fscore, diversity=diversity)

    # return stats.fscore, stats.diversity, pred_li
    return pred_li


def main():
    args = init_helper.get_arguments()

    init_helper.init_logger(args.model_dir, args.log_file)
    init_helper.set_random_seed(args.seed)

    logger.info(vars(args))
    model = get_model(args.model, **vars(args))
    model = model.eval().to(args.device)
    for split_path in args.splits: #different yml file
        split_path = Path(split_path)
        splits = data_helper.load_yaml(split_path)

        # stats = data_helper.AverageMeter('fscore', 'diversity')

        for split_idx, split in enumerate(splits): #different split section in each yml file (0~4)-> 5 section
            ckpt_path = data_helper.get_ckpt_path(args.model_dir, split_path, split_idx)
            state_dict = torch.load(str(ckpt_path),
                                    map_location=lambda storage, loc: storage)
            model.load_state_dict(state_dict)
            ## split = ['test_keys', [video1, video2.....]]
            val_set = data_helper.VideoDataset(split['test_keys'])  #all videos in ith split 
            val_loader = data_helper.DataLoader(val_set, shuffle=False) # val_set = ..videoDataset([path1, path2, path3....])
            # for picks in val_loader:
            #     print(picks)
            # return 
            # fscore, diversity, pred_li = evaluate(model, val_loader, args.nms_thresh, args.device)
            pred_li = evaluate(model, val_loader, args.nms_thresh, args.device)
            
        #     stats.update(fscore=fscore, diversity=diversity)

        #     logger.info(f'{split_path.stem} split {split_idx}: diversity: '
        #                 f'{diversity:.4f}, F-score: {fscore:.4f}')

        # logger.info(f'{split_path.stem}: diversity: {stats.diversity:.4f}, '
        #             f'F-score: {stats.fscore:.4f}')
    print('Length of prediction of all video:', len(pred_li))
    ## Start processing summary video

    # ## Step 1. Build mapping of original videos and summary feature in h5
    # dt = '../mydatasets/vsumm_before_kts.h5'
    # h5in = h5py.File(dt, 'r')
    # dic = {}
    # for video_name, video_file in h5in.items():
    #     # n_frames = video_file['n_frames']
    #     n_frames = video_file['n_frames'][...].astype(int).item()        
    #     # print(video_name, n_frames)
    #     if n_frames not in dic:
    #         dic[n_frames] = [video_name]
    # # for i in dic:
    # #     print(i,dic[i])
    # h5in.close()
    # other = {}
    # dataset_path = '../mydatasets/SUMME_Origin/'
    # for d in os.listdir(dataset_path):
    #     cap = cv2.VideoCapture(os.path.join(dataset_path, d))
    #     count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #     if count in dic:
    #         dic[count].append(d)
    #     else:
    #         if count not in other:
    #             other[count] = d
    #     # 釋放所有資源
    #     cap.release()
    #     cv2.destroyAllWindows()
    
    # # h5out.close()
    # print('Final Mapping:',dic)
    for file_name in pred_li:

        pred_summ = pred_li[file_name]
        cap = cv2.VideoCapture(os.path.join('../mydatasets/video/', file_name+'.mp4'))
        # cap = cv2.VideoCapture('../mydatasets/myvideo/walk_3.mp4')
        width  = int(cap.get(3))
        height = int(cap.get(4))
        fps = int(cap.get(5))
        size = (width, height)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_name =  file_name+'_SUMMARY_OUTPUT'+'.avi'
        output_path = os.path.join('../output/', output_name)
        out = cv2.VideoWriter(output_path, fourcc, fps, size)
        count = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                # 寫入影格
                if pred_summ[count]:
                    out.write(frame)
                count+=1
                    # cv2.imshow('frame',frame)
                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     break
            else:
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    # ## Step 2: Generating Summary Video
    # for index in dic:
    #     if len(dic[index]) < 2:
    #         continue
    #     print('Generating summary video from', dic[index][0], '&', dic[index][1])
    #     pred_summ = pred_li[dic[index][0]]
    #     cap = cv2.VideoCapture(os.path.join(dataset_path, dic[index][1]))
    #     width  = int(cap.get(3))
    #     height = int(cap.get(4))
    #     fps = int(cap.get(5))
    #     size = (width, height)
    #     # print(fps, size)

    #     # 使用 XVID 編碼
    #     fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #     output_name = dic[index][1].split('.')[0] + '_SUMMARY_OUTPUT'+'.avi'
    #     output_path = os.path.join('../output/', output_name)
    #     out = cv2.VideoWriter(output_path, fourcc, fps, size)
    #     count = 0
    #     while(cap.isOpened()):
    #         ret, frame = cap.read()
    #         if ret == True:
    #             # 寫入影格
    #             if pred_summ[count]:
    #                 out.write(frame)
    #             count+=1
    #                 # cv2.imshow('frame',frame)
    #                 # if cv2.waitKey(1) & 0xFF == ord('q'):
    #                 #     break
    #         else:
    #             break
    #     # print('c:',count)
    #     # 釋放所有資源
    #     cap.release()
    #     out.release()
    #     cv2.destroyAllWindows()
    #     print('Finished summary video:', output_path)



if __name__ == '__main__':
    main()
