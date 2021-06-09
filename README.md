This is a modified version of implementing video summarization on custom data.
For basic establishment (eg. pretrain-model, TVSUMM & SUMME datasets, etc), please refer to the DSNet source: https://github.com/li-plus/DSNet

## File structure 
The file structure demonstrated below is an example of my arrangement.
I put custom videos in `mydatasets/video/`, and the generated h5 file in `mydatasets/`.
```
shot-based-video-synopsis
├── mydatasets/
|   ├── video/
|   |   ├──video1.mp4
|   |   └──video2.mp4
|   └── vsumm_custom.h5
├── output/
└── models/
```

## Usage
### Data preparation
```
python data_preparation.py --raw-data /your/video/path --write-h5 /your/h5/path --subsample-size 10
```
### Summary video generation
```
python generate_summary_video.py anchor-based --input-data /your/h5/path --input-video /your/video/path \
                                              --output-folder /your/output/path --model-path /your/model/path 
```