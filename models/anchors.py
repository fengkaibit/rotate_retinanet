import torch
import torch.nn as nn
import numpy as np
from utils.box_utils import xyxy_to_xywh

class Anchors(nn.Module):
    def __init__(self, pyramid_levels=[3, 4, 5, 6, 7],
                 strides=None, sizes=None, ratios=None, scales=None, angles=None):
        super(Anchors, self).__init__()

        self.pyramid_levels = pyramid_levels
        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]
        else:
            self.strides = strides
        if sizes is None:
            self.sizes = [2 ** (x+2) for x in self.pyramid_levels]
        else:
            self.sizes = sizes
        if ratios is None:
            self.ratios = np.array([0.5, 1, 2])
        else:
            self.ratios = ratios
        if scales is None:
            self.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
        else:
            self.scales = scales
        if angles is None:
            self.angles = np.array([-90, -60, -30])
        else:
            self.angles = angles


    def forward(self, batch_imgs):
        img_shape = batch_imgs.shape[2:]
        img_shape = np.array(img_shape)
        img_shapes = [(img_shape + 2 ** x -1) // (2 ** x) for x in self.pyramid_levels]

        all_anchors = np.zeros((0, 5)).astype(np.float32)
        for idx, p in enumerate(self.pyramid_levels):
            anchors = generate_anchors(base_size=self.sizes[idx], ratios=self.ratios, scales=self.scales, angles=self.angles)
            shift_anchors = shift(img_shapes[idx], self.strides[idx], anchors)
            all_anchors = np.vstack((all_anchors, shift_anchors))
        all_anchors = xyxy_to_xywh(all_anchors)
        #all_theta = np.array([-90] * all_anchors.shape[0]).astype(np.float32)
        #all_theta = np.expand_dims(all_theta, axis=0).T
        #all_anchors = np.hstack((all_anchors, all_theta))
        all_anchors = np.expand_dims(all_anchors, axis=0)

        if torch.cuda.is_available():
            return torch.from_numpy(all_anchors.astype(np.float32)).cuda()
        else:
            return torch.from_numpy(all_anchors.astype(np.float32))

def generate_anchors(base_size=16, ratios=None, scales=None, angles=None):
    if ratios is None:
        ratios = np.array([0.5, 1, 2])
    if scales is None:
        scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
    if angles is None:
        angles = np.array([-90, -60, -30])

    num_anchors = len(ratios) * len(scales) * len(angles)
    anchors = np.zeros((num_anchors, 4))
    thetas = np.zeros((num_anchors, 1))

    thetas[:, 0] = np.tile(angles, (len(scales), len(ratios))).T.reshape(-1)
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios) * len(angles))).T
    areas = anchors[:, 2] * anchors[:, 3]
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales) * len(angles)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales) * len(angles))

    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    anchors = np.hstack((anchors, thetas))

    # anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T
    # areas = anchors[:, 2] * anchors[:, 3]
    #
    # anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    # anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))
    #
    # anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    # anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors

def shift(shape, stride, anchors):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()

    thetas = np.zeros((shifts.shape[0], 1))
    shifts = np.hstack((shifts, thetas))

    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = anchors.reshape((1, A, 5)) + shifts.reshape((1, K, 5)).transpose((1, 0, 2))
    all_anchors = all_anchors.reshape((K * A, 5))

    return all_anchors


if __name__ == '__main__':
    a = Anchors()
    x = torch.ones([1, 3, 800, 1280])
    anchor = a(x)
    print(anchor.shape)