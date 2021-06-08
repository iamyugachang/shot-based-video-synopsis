import torch
from torchvision.models.googlenet import GoogLeNet 
from torchvision.models.resnet import ResNet
import cv2
import os
from torchvision.models import googlenet
from torchvision.models import resnet50
import torchvision.transforms.functional as TF
from tqdm import tqdm
import numpy as np
from torch.autograd import Variable
# from numpy import save

def extract_features_googlenet(self, inputs):
        # Returns output of the last three layers (before softmax, after dropout)
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

def extract_features_resnet(self, inputs):
        # See note [TorchScript super()]
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


# @profile
def extract_feature(video_path, model_name, pick_gap): #-> return feature
    print('Start extracting feature from "{}", using "{}"'.format(video_path, model_name))
    GoogLeNet.extract_features_googlenet = extract_features_googlenet #do unbound method
    # ResNet.extract_features_resnet = extract_features_resnet #do unbound method
    model = googlenet(pretrained=True)
    # model = resnet50(pretrained=True)
    model = model.cuda() #Using GPU
    feature_stack = []
    cap = cv2.VideoCapture(video_path)
    n_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # while(cap.isOpened()):
    # saved_tensor_path = 'saved_tensor/'
    # flag = True
    
    for count in tqdm(range(n_frame)):
        ret, frame = cap.read()
        if ret == True:
            if count%pick_gap == 0:
                # print('Now processing {}-{} th frame'.format(count, count+pick_gap))
                image_tensor = TF.to_tensor(frame).unsqueeze(0)
                image_tensor = Variable(image_tensor).cuda() #Using GPU
                feature = model.extract_features_googlenet(image_tensor)
                # feature = model.extract_features_resnet(image_tensor)
                if count == 0:
                    # flag = False
                    # feature = model.extract_features(image_tensor)
                    npary_set = feature.cpu().detach().numpy()
                else:
                    # feature = torch.cat([feature, model.extract_features(image_tensor)],0)
                    npary_set = np.concatenate((npary_set, feature.cpu().detach().numpy()), axis=0)
                del image_tensor, feature
                # if feature.shape[0]%350 == 0:
                #     flag = True
                #     torch.save(feature, os.path.join(saved_tensor_path, str(count)+'.pt')) #save tensor
        else:
            break
    cap.release()
    # return feature.detach().numpy()
    return npary_set


if __name__ == '__main__':
    feature_stack = extract_feature('../mydatasets/video/walk_3.mp4', 'googlenet', 15)
    print(feature_stack)
    print(feature_stack.shape)
