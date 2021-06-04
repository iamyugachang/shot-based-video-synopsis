import cv2
import torchvision.transforms.functional as TF
video_path = '../datasets/test.mp4'
cap = cv2.VideoCapture(video_path)
_, frame = cap.read()
x = TF.to_tensor(frame)
print(x.shape)