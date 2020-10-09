import torch
import cv2
import random
import numpy as np
import os
from dataset.voc_dataset import UCAS_AOD_CLASSES
from models.retinanet import build_retinanet
from utils.box_utils import center_to_points8

def detect_image(model_path):
    retinanet = build_retinanet()
    checkpoint = torch.load(model_path)
    retinanet.load_state_dict(checkpoint['state_dict'])

    if torch.cuda.is_available():
        retinanet = retinanet.cuda()

    retinanet.training = False
    retinanet.eval()

    with open('/home/fengkai/datasets/UCAS-AOD/test_list.txt') as f:
        line = f.readlines()
        for l in line:
            image_path = os.path.join('/home/fengkai/datasets/UCAS-AOD/JPEGImages', l.strip())
            image = cv2.imread(image_path)
            image_orig = image.copy()
            size = (1280, 800)

            interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
            interp_method = interp_methods[random.randrange(5)]
            image = cv2.resize(image, size, interpolation=interp_method)
            image = image.astype(np.float32)
            rgb_mean = (104, 117, 123)
            image -= rgb_mean
            image = image.transpose(2, 0, 1)
            image = np.expand_dims(image, 0)

            with torch.no_grad():
                image = torch.from_numpy(image)
                if torch.cuda.is_available():
                    image = image.cuda()
                scores, classification, transformed_anchors = retinanet(image.float())
                idxs = np.where(scores.cpu() > 0.5)

                for j in range(idxs[0].shape[0]):
                    bbox = transformed_anchors[idxs[0][j], :]
                    scale_x = size[0] / image_orig.shape[1]
                    scale_y = size[1] / image_orig.shape[0]
                    bbox = torch.unsqueeze(bbox, dim=0).cpu().numpy()

                    bbox = center_to_points8(bbox)
                    bbox[:,0::2] /= scale_x
                    bbox[:,1::2] /= scale_y
                    bbox = bbox[0]
                    label_name = UCAS_AOD_CLASSES[int(classification[idxs[0][j]])]
                    #score = scores[j]
                    score = scores[idxs[0][j]]
                    caption = '{} {:.3f}'.format(label_name, score)
                    pt1 = (int(bbox[0]), int(bbox[1]))
                    pt2 = (int(bbox[2]), int(bbox[3]))
                    pt3 = (int(bbox[4]), int(bbox[5]))
                    pt4 = (int(bbox[6]), int(bbox[7]))
                    center = ((pt1[0] + pt3[0]) // 2, (pt1[1] + pt3[1]) // 2)
                    cv2.putText(image_orig, caption, center, 1, 1, (0, 255, 0))
                    cv2.line(image_orig, pt1, pt2, (0, 255, 0), 2)
                    cv2.line(image_orig, pt2, pt3, (0, 255, 0), 2)
                    cv2.line(image_orig, pt3, pt4, (0, 255, 0), 2)
                    cv2.line(image_orig, pt4, pt1, (0, 255, 0), 2)

                cv2.imshow('detections', image_orig)
                cv2.waitKey(0)

if __name__ == '__main__':
    model_path = '/home/fengkai/PycharmProjects/my_retinanet_rotate/10.8/model_final.pth'
    detect_image(model_path)
