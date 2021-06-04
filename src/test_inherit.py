from  torchvision.models.googlenet import GoogLeNet
import torchvision
import torch.nn as nn
import cv2
import torch
import torchvision.transforms.functional as TF

def extract_features(self, inputs):
        """ Returns output of the final convolution layer """
        x = self.conv1(inputs)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.dropout(x)
        return x


if __name__ == '__main__':
    cap = cv2.VideoCapture('../mydatasets/video/walk_3.mp4')
    _, frame = cap.read()
    image_tensor = TF.to_tensor(frame).unsqueeze(0)
    GoogLeNet.extract_features = extract_features
    model = torchvision.models.googlenet(pretrained=True)
    feature = model.extract_features(image_tensor)
    print(feature.shape)

