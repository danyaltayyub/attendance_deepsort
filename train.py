import numpy as np
import torch
import cv2
from deep_sort.sort.detection import Detection
from deep_sort.sort.preprocessing import non_max_suppression
from utils_ds.parser import get_config
from facenet_pytorch import MTCNN
from os import listdir
from deep_sort.deep.extractor import Extractor
from deep_sort import DeepSortFace, build_tracker


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
conf = get_config()
conf.merge_from_file("/home/transdata/DeepSORT_Face/configs/deep_sort.yaml")
ds = build_tracker(conf, use_cuda=True)

detector = MTCNN(keep_all=True, device=device)



# def _xywh_to_xyxy(bbox_xywh, width , height):
#         x, y, w, h = bbox_xywh      # xc, yc, w, h
#         x1 = max(int(x - w / 2), 0)
#         x2 = min(int(x + w / 2), width - 1)
#         y1 = max(int(y - h / 2), 0)
#         y2 = min(int(y + h / 2), height - 1)
#         return x1, y1, x2, y2

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y
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
        ds.height, ds.width,_ = im0.shape
        img = cv2.resize(im0, (ds.width // 2, ds.height // 2)) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

        
        with torch.no_grad():
            boxes , probs = detector.detect(img)

        print(boxes)

        if boxes is not None and len(boxes):
            boxes = boxes * 2      # x1,y1,x2,y2  go back to original image

            bbox_xywh = xyxy2xywh(boxes)    # (#obj, 4)     xc,yc,w,h

            # add margin here. only need to revise width and height
            bbox_xywh[:, 2:] = bbox_xywh[:, 2:] * (1 + 0.2)

            ds.height, ds.width = im0.shape[:2]
            features = ds._get_features(bbox_xywh, im0)


            embedding_list.append(features)
            name_list.append(ls)


data = [embedding_list,name_list]
torch.save(data, 'data.pt') # saving data.pt file
print("\n✔ Training done.")
print(f"✔ Total Persons in our Dataset: {len(name_list)}")
for p in name_list:
    print(f"\t\t✅{p}")

print(type(embedding_list))
