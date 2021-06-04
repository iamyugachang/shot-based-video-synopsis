# import torch
# import cv2
# from googlenet_pytorch import GoogLeNet 
# import torchvision.transforms.functional as TF
# model = GoogLeNet.from_pretrained('googlenet')
# video_path = '../mydatasets/Jumps.mp4'
# cap = cv2.VideoCapture(video_path)
# _, single_frame = cap.read()
# image_tensor = TF.to_tensor(single_frame).unsqueeze(0)
# feature = torch.flatten(model.extract_features_avg(image_tensor), 0)
# print(feature)

import torchvision
import cv2
import torchvision.transforms.functional as TF
model = torchvision.models.resnet50(pretrained=True)
video_path = '../mydatasets/Jumps.mp4'
cap = cv2.VideoCapture(video_path)
_, single_frame = cap.read()
image_tensor = TF.to_tensor(single_frame).unsqueeze(0)
print(model.extract_features(image_tensor).shape)