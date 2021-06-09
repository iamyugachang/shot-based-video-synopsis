from create_h5 import create_h5
from make_shots import make_shots
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-video', type=str, default='../mydatasets/video')
    parser.add_argument('--write-h5', type=str, default='../mydatasets/vsumm_custom.h5')
    parser.add_argument('--subsample-size', type=int, default=10)
    args = parser.parse_args()
    middle_file = args.write_h5+'_middle.h5' #A middle file of initialization of the dataset
    create_h5(args.raw_video, middle_file, args.subsample_size)
    make_shots(middle_file, args.write_h5, args.subsample_size)