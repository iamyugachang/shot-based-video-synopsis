import argparse
import h5py
import numpy as np
from kts.cpd_auto import cpd_auto

def make_shots(h5in, h5out, subsample_size):
    h5in = h5py.File(h5in, 'r')
    h5out = h5py.File(h5out, 'w')

    for video_name, video_file in h5in.items():
        features = video_file['features'][...].astype(np.float32)
        gtscore = video_file['gtscore'][...].astype(np.float32)
        gtsummary = video_file['gtsummary'][...].astype(np.float32)

        seq_len = features.shape[0]
        n_frames = seq_len * subsample_size - 1
        picks = np.arange(0, seq_len) * subsample_size
        kernel = np.matmul(features, features.T)
        change_points, _ = cpd_auto(kernel, seq_len - 1, 1)
        change_points *= subsample_size
        change_points = np.hstack((0, change_points, n_frames))
        begin_frames = change_points[:-1]
        end_frames = change_points[1:]
        change_points = np.vstack((begin_frames, end_frames - 1)).T

        n_frame_per_seg = end_frames - begin_frames

        h5out.create_dataset(video_name + '/features', data=features)
        h5out.create_dataset(video_name + '/gtscore', data=gtscore)
        # h5out.create_dataset(name + '/user_summary', data=data_of_name)
        h5out.create_dataset(video_name + '/change_points', data=change_points)
        h5out.create_dataset(video_name + '/n_frame_per_seg', data=n_frame_per_seg)
        h5out.create_dataset(video_name + '/n_frames', data=n_frames)
        h5out.create_dataset(video_name + '/picks', data=picks)
        # h5out.create_dataset(video_name + '/n_steps', data=data_of_name)
        h5out.create_dataset(video_name + '/gtsummary', data=gtsummary)
        h5out.create_dataset(video_name + '/video_name', data=video_name)

    h5in.close()
    h5out.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--read-h5', type=str, default='../mydatasets/vsumm_custom.h5')
    parser.add_argument('--write-h5', type=str, default='../mydatasets/vsumm_custom_after_kts.h5')
    parser.add_argument('--subsample-size', type=int, default=10)
    args = parser.parse_args()
    make_shots(args.read_h5, args.write_h5, args.subsample_size)
