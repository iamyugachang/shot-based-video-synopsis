import torch
from googlenet_pytorch import GoogLeNet 
import cv2
import torchvision
import torchvision.transforms.functional as TF


def extract_feature(video_path, model_name, pick_gap): #-> return feature
    print('Start extracting feature from "{}", using "{}"'.format(video_path, model_name))
    # device = torch.device("cuda")
    # model = GoogLeNet.from_pretrained(model_name)
    model = torchvision.models.googlenet(pretrained=True)
    # model = torchvision.models.resnet50(pretrained=True)
    # model.to_device('gpu0')
    feature_stack = []
    cap = cv2.VideoCapture(video_path)
    count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            if count%pick_gap == 0:
                print('Now processing {}-{} th frame'.format(count, count+pick_gap))
                image_tensor = TF.to_tensor(frame).unsqueeze(0)
                # feature = model.extract_features_avg(image_tensor).squeeze(2).squeeze(2).squeeze(0)
                # feature = torch.flatten(model.extract_features_avg(image_tensor), 0)
                # feature = torch.flatten(model.extract_features(image_tensor), 0)
                if count == 0:
                    feature = model.extract_features(image_tensor)
                else:
                    feature = torch.cat([feature, model.extract_features(image_tensor)],0)   
        else:
            break
        count+=1
    cap.release()
    return feature.detach().numpy()


if __name__ == '__main__':
    feature_stack = extract_feature('../mydatasets/walk_3.mp4', 'googlenet', 15)
    print(feature_stack)
    print(feature_stack.shape)
