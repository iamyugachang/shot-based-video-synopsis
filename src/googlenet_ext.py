import torch
from  torchvision.models.googlenet import GoogLeNet 
import cv2
from torchvision.models import googlenet
import torchvision.transforms.functional as TF
from tqdm import tqdm
# from torch.autograd import Variable
# from numpy import save

def extract_features(self, inputs):
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


def extract_feature(video_path, model_name, pick_gap): #-> return feature
    print('Start extracting feature from "{}", using "{}"'.format(video_path, model_name))
    GoogLeNet.extract_features = extract_features #do unbound method
    model = googlenet(pretrained=True)
    # model = torchvision.models.resnet50(pretrained=True)
    # model = model.cuda() #Using GPU
    feature_stack = []
    cap = cv2.VideoCapture(video_path)
    n_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # while(cap.isOpened()):
    for count in tqdm(range(n_frame)):
        ret, frame = cap.read()
        if ret == True:
            if count%pick_gap == 0:
                # print('Now processing {}-{} th frame'.format(count, count+pick_gap))
                image_tensor = TF.to_tensor(frame).unsqueeze(0)
                # image_tensor = Variable(image_tensor).cuda() #Using GPU
                if count == 0:
                    feature = model.extract_features(image_tensor)
                    # torch.save(feature, 'save_tensor.pt') #save tensor
                else:
                    feature = torch.cat([feature, model.extract_features(image_tensor)],0)   
        else:
            break
    cap.release()
    return feature.detach().numpy()


if __name__ == '__main__':
    feature_stack = extract_feature('../mydatasets/video/walk_3.mp4', 'googlenet', 15)
    print(feature_stack)
    print(feature_stack.shape)
