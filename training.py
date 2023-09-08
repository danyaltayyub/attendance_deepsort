import cv2
import torch
import numpy as np
from os import listdir


class Training():
    def __init__(self, deepsort_obj, detector, save_data, scale , margin_ratio):
        self.ds = deepsort_obj
        self.detector = detector
        self.save_data = save_data
        self.scale = scale
        self.margin_ratio = margin_ratio

    def initiate_training(self):
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
                self.ds.height, self.ds.width,_ = im0.shape
                img = cv2.resize(im0, (self.ds.width // self.scale, self.ds.height // self.scale)) 
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

                
                with torch.no_grad():
                    boxes , _ = self.detector.detect(img)


                if boxes is not None and len(boxes):
                    boxes = boxes * self.scale      # x1,y1,x2,y2  go back to original image

                    bbox_xywh = self.xyxy2xywh(boxes)    # (#obj, 4)     xc,yc,w,h

                    # add margin here. only need to revise width and height
                    bbox_xywh[:, 2:] = bbox_xywh[:, 2:] * (1 + self.margin_ratio)

                    self.ds.height, self.ds.width = im0.shape[:2]
                    features = self.ds._get_features(bbox_xywh, im0)


                    embedding_list.append(features)
                    name_list.append(ls)


        data = [embedding_list,name_list]
        torch.save(data, 'data.pt') # saving data.pt file
        print("\n✔ Training done.")
        print(f"✔ Total Persons in our Dataset: {len(name_list)}")
        for p in name_list:
            print(f"\t\t✅{p}")


    def xyxy2xywh(self, x):
        # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
        y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
        y[:, 2] = x[:, 2] - x[:, 0]  # width
        y[:, 3] = x[:, 3] - x[:, 1]  # height
        return y