from utils.cython_utils.cython_nms import nms as _nms
from utils.rotate_cython_utils.rotate_polygon_nms import rotate_gpu_nms
import torch
import numpy as np

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

def nms(bboxes, scores, iou_thresh, max_ouput_size=None):
    # bboxes: [xmin, ymin, xmax, ymax]
    scores = np.expand_dims(scores, 1)
    det = np.hstack((bboxes, scores))
    keep = _nms(det, iou_thresh)
    if max_ouput_size is not None:
        keep = keep[:max_ouput_size]
    return keep

def rotate_nms(bboxes, scores, iou_thresh, gpu_id=0):
    # bboxes: [x, y, w, h, theta]
    bboxes = tonumpy(bboxes)
    scores = tonumpy(scores)
    scores = np.expand_dims(scores, 1)
    det = np.hstack((bboxes, scores))
    keep = rotate_gpu_nms(det, iou_thresh, gpu_id)
    return keep


if __name__ == '__main__':
    box = np.array([[100, 100, 150, 150], [100, 100, 150, 150], [200, 200, 250, 250]],
                   dtype=np.float32)
    score = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    print(nms(box, score, 0.7))

    boxes = np.array([[50, 50, 100, 100, 30],
                      [50, 50, 100, 100, 30],
                      [50, 50, 100, 100, -45.],
                      [200, 200, 100, 100, 0.]], dtype=np.float32)

    scores = np.array([0.99, 0.88, 0.66, 0.77], dtype=np.float32)
    keep = rotate_nms(boxes, scores, 0.5)
    print(keep)
