from utils.cython_utils.cython_bbox import bbox_overlaps
from utils.rotate_cython_utils.rbbox_overlaps import rbbx_overlaps
import numpy as np
import torch

def tonumpy(data):
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()

def totensor(data, cuda=True):
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    if isinstance(data, torch.Tensor):
        tensor = data.detach()
    if cuda:
        tensor = tensor.cuda()
    return tensor

def bbox_iou(bbox_a, bbox_b):
    # [x1, y1, x2, y2]
    return bbox_overlaps(bbox_a, bbox_b)

def bbox_iou_tensor(bbox_a, bbox_b):
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError

    A = bbox_a.shape[0]
    B = bbox_b.shape[0]
    max_xy = torch.min(bbox_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       bbox_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(bbox_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       bbox_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    inter = inter[:, :, 0] * inter[:, :, 1]

    area_a = ((bbox_a[:, 2] - bbox_a[:, 0]) * (bbox_a[:, 3] - bbox_a[:, 1])).unsqueeze(1).expand_as(inter)
    area_b = ((bbox_b[:, 2] - bbox_b[:, 0]) * (bbox_b[:, 3] - bbox_b[:, 1])).unsqueeze(0).expand_as(inter)
    union = area_a + area_b - inter
    return inter / union

def rotate_bbox_iou(bbox_a, bbox_b):
    #[x1, y1, w, h, theta]
    bbox_a = tonumpy(bbox_a)
    bbox_b = tonumpy(bbox_b)
    iou = rbbx_overlaps(bbox_a, bbox_b)
    return totensor(iou)


if __name__ == '__main__':
    boxes1 = np.array([[50, 50, 100, 300], [60, 60, 100, 200]], np.float)
    boxes2 = np.array([[50, 50, 100, 300], [200, 200, 100, 200]], np.float)
    r_boxes1 = np.array([[50, 50, 100, 100, 0],
                       [60, 60, 100, 200, -60]], np.float32)
    r_boxes2 = np.array([[50, 50, 100, 100, -45.],
                       [200, 200, 100, 200, -70]], np.float32)
    print(bbox_iou(boxes1, boxes2))
    print(bbox_iou_tensor(torch.from_numpy(boxes1), torch.from_numpy(boxes2)))
    print(rotate_bbox_iou(r_boxes1, r_boxes2))