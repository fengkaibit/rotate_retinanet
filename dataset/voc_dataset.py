import torch
import torch.utils.data as data
import cv2
import os
import xml.etree.ElementTree as ET
import numpy as np
from utils.box_utils import target_to_center

UCAS_AOD_CLASSES = ('plane', 'car')
NAME_LABEL_MAP = {'plane': 0, 'car': 1}

class AnnotationTransform(object):
    def __init__(self, class_to_ind=None, keep_difficult=True):
        self.class_to_ind = class_to_ind or dict(
            zip(UCAS_AOD_CLASSES, range(len(UCAS_AOD_CLASSES)))
        )
        self.keep_difficult = keep_difficult

    def __call__(self, target):
        res = np.empty((0, 9))
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(float(bbox.find(pt).text))
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res = np.vstack((res, bndbox))
        return res

class VOCDataset(data.Dataset):
    def __init__(self, root, preprocess=None, target_transform=None):
        super(VOCDataset, self).__init__()
        self.root = root
        self.preprocess = preprocess
        self.target_transform = target_transform
        self.anno_path = os.path.join(self.root, 'Annotations', '%s')
        self.img_path = os.path.join(self.root, 'JPEGImages', '%s')
        self.ids = list()
        with open(os.path.join(self.root, 'train_list.txt'), 'r') as f:
            self.ids = [tuple(line.split()) for line in f]

    def __getitem__(self, index):
        img_id = self.ids[index]
        target = ET.parse(self.anno_path %img_id[0].replace('.png', '.xml')).getroot()
        img = cv2.imread(self.img_path % img_id[0])
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.preprocess is not None:
            img, target = self.preprocess(img, target)

        target = target_to_center(target)

        return torch.from_numpy(img), target
        #return img, target

    def __len__(self):
        return len(self.ids)

def detection_collate(batch):
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                anno = torch.from_numpy(tup).float()
                targets.append(anno)
    return (torch.stack(imgs, 0), targets)

if __name__ == '__main__':
    root = '/home/fengkai/datasets/UCAS-AOD/'
    from dataset.data_augment import PreProcess
    a = VOCDataset(root, preprocess=PreProcess(), target_transform=AnnotationTransform())
    import random
    idx = random.randint(0, len(a))
    img, target = a.__getitem__(idx)
    img = np.array(img.transpose(1, 2, 0), dtype=np.int8)
    for t in target:
        x_c, y_c = t[0], t[1]
        w, h, theta = t[2], t[3], t[4]
        name = UCAS_AOD_CLASSES[int(t[5])]
        rect = ((x_c, y_c), (w, h), theta)
        rect = cv2.boxPoints(rect)
        rect = np.int0(rect)
        cv2.drawContours(img, [rect], -1, (255, 0, 0))
        cv2.putText(img, name, (int(x_c), int(y_c)), 1, 1, (0, 255, 0))
    cv2.imshow('src', img)
    cv2.waitKey(0)
