import cv2
import numpy as np
import torch
import torch.nn as nn
import math

def tonumpy(data):
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()

def target_to_center(target):
    """
    :param target: [x1, y1, x2, y2, x3, y3, x4, y4, label]
    :return: [x_c, y_c, w, h, theta, label]
    """
    bboxes = np.empty((0, 6))
    for rect in target:
        bbox = np.int0(rect[:-1])
        lable = rect[-1]
        bbox = bbox.reshape((4, 2))
        rect1 = cv2.minAreaRect(bbox)
        x_c, y_c, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]
        if abs(theta - 0) < 1e-7:
            w, h = h, w
            theta -= 90
        bboxes = np.vstack((bboxes, [x_c, y_c, w, h, theta, lable]))

    return bboxes

def points8_to_center(points_coordinate):
    """
    :param points_coordinate: [x1, y1, x2, y2, x3, y3, x4, y4]
    :return: [x_c, y_c, w, h, theta]
    """
    bboxes = np.empty((0, 5))
    for rect in points_coordinate:
        bbox = np.int0(rect)
        bbox = bbox.reshape((4, 2))
        rect1 = cv2.minAreaRect(bbox)
        x_c, y_c, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]
        if abs(theta - 0) < 1e-7:
            w, h = h, w
            theta -= 90
        bboxes = np.vstack((bboxes, [x_c, y_c, w, h, theta]))

    return bboxes

def points8_to_points4(points_coordinate):
    """
    :param points_coordinate: [x1, y1, x2, y2, x3, y3, x4, y4]
    :return: [xmin, ymin, xmax, ymax]
    """
    bboxes = np.empty((0, 4))
    for rect in points_coordinate:
        x = rect[0::2]
        y = rect[1::2]
        xmin, xmax = min(x), max(x)
        ymin, ymax = min(y), max(y)
        bboxes = np.vstack((bboxes, [xmin, ymin, xmax, ymax]))

    return bboxes

def center_to_points8(center_coordinate):
    #center_coordinate = tonumpy(center_coordinate)
    bboxes = np.empty((0, 8))
    for rect in center_coordinate:
        bbox = cv2.boxPoints(((rect[0], rect[1]), (rect[2], rect[3]), rect[4]))
        bbox = np.reshape(bbox, [-1, ])
        bboxes = np.vstack((bboxes, [bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5], bbox[6], bbox[7]]))

    return bboxes

def xyxy_to_xywh(points_coordinate):
    """
    :param points_coordinate: [xmin, ymin, xmax, ymax]
    :return: [x_c, y_c, w, h]
    """
    # x = points_coordinate[:, 0::2]
    # y = points_coordinate[:, 1::2]
    #
    # x_c = (x[:, 1] + x[:, 0]) * 0.5
    # y_c = (y[:, 1] + y[:, 0]) * 0.5
    # w = x[:, 1] - x[:, 0]
    # h = y[:, 1] - y[:, 0]
    # x_c = np.expand_dims(x_c, axis=0).T
    # y_c = np.expand_dims(y_c, axis=0).T
    # w = np.expand_dims(w, axis=0).T
    # h = np.expand_dims(h, axis=0).T
    #
    # bboxes = np.hstack((x_c, y_c, w, h))

    x_c = (points_coordinate[:, 2] + points_coordinate[:, 0]) * 0.5
    y_c = (points_coordinate[:, 3] + points_coordinate[:, 1]) * 0.5
    w = points_coordinate[:, 2] - points_coordinate[:, 0]
    h = points_coordinate[:, 3] - points_coordinate[:, 1]
    theta = points_coordinate[:, 4]
    x_c = np.expand_dims(x_c, axis=0).T
    y_c = np.expand_dims(y_c, axis=0).T
    w = np.expand_dims(w, axis=0).T
    h = np.expand_dims(h, axis=0).T
    theta = np.expand_dims(theta, axis=0).T

    bboxes = np.hstack((x_c, y_c, w, h, theta))

    return bboxes

class BBoxTransform(nn.Module):
    def __init__(self, mean=None, std=None):
        super(BBoxTransform, self).__init__()
        if mean is None:
            if torch.cuda.is_available():
                self.mean = torch.from_numpy(np.array([0, 0, 0, 0, 0]).astype(np.float32)).cuda()
            else:
                self.mean = torch.from_numpy(np.array([0, 0, 0, 0, 0]).astype(np.float32))
        else:
            self.mean = mean

        if std is None:
            if torch.cuda.is_available():
                self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2, 0.1]).astype(np.float32)).cuda()
            else:
                self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2, 0.1]).astype(np.float32))
        else:
            self.std = std

    def forward(self, boxes, deltas):
        widths = boxes[:, :, 2]
        heights = boxes[:, :, 3]
        ctr_x = boxes[:, :, 0]
        ctr_y = boxes[:, :, 1]
        theta = boxes[:, :, 4]

        dx = deltas[:, :, 0] * self.std[0] + self.mean[0]
        dy = deltas[:, :, 1] * self.std[1] + self.mean[1]
        dw = deltas[:, :, 2] * self.std[2] + self.mean[2]
        dh = deltas[:, :, 3] * self.std[3] + self.mean[3]
        dtheta = deltas[:, :, 4] * self.std[4] + self.mean[4]

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w     = torch.exp(dw) * widths
        pred_h     = torch.exp(dh) * heights
        pred_theta = theta + dtheta * 180.0 / math.pi

        pred_boxes = torch.stack([pred_ctr_x, pred_ctr_y, pred_w, pred_h, pred_theta], dim=2)

        return pred_boxes

class ClipBoxes(nn.Module):

    def __init__(self):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):
        batch_size, num_channels, height, width = img.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0, max=width)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0, max=height)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)

        return boxes


if __name__ == '__main__':
    root = '/home/fengkai/datasets/UCAS-AOD/'
    from dataset.data_augment import PreProcess
    from dataset.voc_dataset import VOCDataset, AnnotationTransform, UCAS_AOD_CLASSES

    a = VOCDataset(root, preprocess=PreProcess(), target_transform=AnnotationTransform())
    import random

    idx = random.randint(0, len(a))
    img, target = a.__getitem__(idx)
    img = np.array(img.transpose(1, 2, 0), dtype=np.int8)

    bbox = target[:, :-1]
    name = target[:, -1]

    bbox = points8_to_center(bbox)
    print(bbox)
    for k in bbox:
        x_c, y_c, w, h, theta = k[0], k[1], k[2], k[3], k[4]
        rect = ((x_c, y_c), (w, h), theta)
        rect = cv2.boxPoints(rect)
        rect = np.int0(rect)
        cv2.drawContours(img, [rect], -1, (255, 0, 0), 2)

    bbox = center_to_points8(bbox)
    bbox = np.int0(bbox)

    for t in bbox:
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))
        pt4 = (int(t[6]), int(t[7]))
        #cv2.line(img, pt1, pt2, (0, 0, 255), 2)
        #cv2.line(img, pt2, pt3, (0, 0, 255), 2)
        #cv2.line(img, pt3, pt4, (0, 0, 255), 2)
        #cv2.line(img, pt4, pt1, (0, 0, 255), 2)
        #name = UCAS_AOD_CLASSES[int(t[8])]
        #cv2.putText(img, name, pt1, 1, 1, (0, 255, 0))
    cv2.imshow('src', img)
    cv2.waitKey(0)
