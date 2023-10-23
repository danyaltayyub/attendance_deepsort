import cv2
import torch
import numpy as np
from os import listdir
from facenet_pytorch import MTCNN
from deep_sort.deep.extractor import Extractor

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
detector = MTCNN(keep_all=True, device=device)
extractor = Extractor(use_cuda=True)

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def _xywh_to_xyxy(bbox_xywh, width, height):
    x, y, w, h = bbox_xywh      # xc, yc, w, h
    x1 = max(int(x - w / 2), 0)
    x2 = min(int(x + w / 2), width - 1)
    y1 = max(int(y - h / 2), 0)
    y2 = min(int(y + h / 2), height - 1)
    return x1, y1, x2, y2


def _get_features(bbox_xywh, ori_img, width, height):
    """
    :param bbox_xywh:
    :param ori_img: cv2 array (h,w,3)
    :return:
    """
    im_crops = []
    for box in bbox_xywh:
        x1, y1, x2, y2 = _xywh_to_xyxy(box, width, height)
        im = ori_img[y1:y2, x1:x2]
        im_crops.append(im)
    if im_crops:
        features = extractor(im_crops)
    else:
        features = np.array([])
    return features

name_list = [] # list of names corrospoing to cropped photos
embedding_list = [] # list of embeding matrix after conversion from cropped faces to embedding matrix using resnet
database = "./saved/"
embedding_list = []
name_list = []

for ls in listdir(database):
    subdir = database + '/' + ls
    # temp_list = []

    for subls in listdir(subdir):
        im0 = cv2.imread(subdir + '/' + subls)
        height, width,_ = im0.shape
        img = cv2.resize(im0, (width // 2, height // 2)) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

        
        with torch.no_grad():
            boxes , _ = detector.detect(img)


        if boxes is not None and len(boxes):
            boxes = boxes * 2     # x1,y1,x2,y2  go back to original image

            bbox_xywh = xyxy2xywh(boxes)    # (#obj, 4)     xc,yc,w,h

            # add margin here. only need to revise width and height
            bbox_xywh[:, 2:] = bbox_xywh[:, 2:] * (1 + 0.2)

            features = _get_features(bbox_xywh, im0, width, height)


            embedding_list.append(features)
            name_list.append(ls)


data = [embedding_list,name_list]
print("Embeddingssssssssssss", embedding_list[0])
torch.save(data, 'data2.pt') # saving data.pt file
print("\n✔ Training done.")
print(f"✔ Total Persons in our Dataset: {len(name_list)}")
for p in name_list:
    print(f"\t\t✅{p}")