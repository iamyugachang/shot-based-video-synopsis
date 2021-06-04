import argparse

import h5py
import numpy as np

from kts.cpd_auto import cpd_auto


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()

    h5in = h5py.File(args.dataset, 'r')
    h5out = h5py.File(args.dataset + '.custom.h5', 'w')

    for video_name, video_file in h5in.items():
        features = video_file['features'][...].astype(np.float32)
        gtscore = video_file['gtscore'][...].astype(np.float32)
        gtsummary = video_file['gtsummary'][...].astype(np.float32)

        seq_len = gtscore.size
        print('seq_length:', seq_len)
        n_frames = seq_len * 15 - 1
        print('n_frames', n_frames)
        picks = np.arange(0, seq_len) * 15
        print('length of picks:', len(picks))
        kernel = np.matmul(features, features.T)
        print('length of kernel:', kernel)
        change_points, _ = cpd_auto(kernel, seq_len - 1, 1)
        print('1st cp:', change_points)
        change_points *= 15
        print('2nd cp:', change_points)
        change_points = np.hstack((0, change_points, n_frames))
        print('3rd cp:', change_points)
        begin_frames = change_points[:-1]
        end_frames = change_points[1:]
        change_points = np.vstack((begin_frames, end_frames - 1)).T
        print('4th cp:', change_points)

        n_frame_per_seg = end_frames - begin_frames
        print('n_frame_per_seq:', n_frame_per_seg)

        h5out.create_dataset(video_name + '/features', data=features)
        h5out.create_dataset(video_name + '/gtscore', data=gtscore)
        # h5out.create_dataset(name + '/user_summary', data=data_of_name)
        h5out.create_dataset(video_name + '/change_points',
                             data=change_points)
        h5out.create_dataset(video_name + '/n_frame_per_seg',
                             data=n_frame_per_seg)
        h5out.create_dataset(video_name + '/n_frames', data=n_frames)
        h5out.create_dataset(video_name + '/picks', data=picks)
        # h5out.create_dataset(video_name + '/n_steps', data=data_of_name)
        h5out.create_dataset(video_name + '/gtsummary', data=gtsummary)
        # h5out.create_dataset(name + '/video_name', data=data_of_name)

    h5in.close()
    h5out.close()


if __name__ == '__main__':
    main()
