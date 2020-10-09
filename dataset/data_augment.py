import cv2
import random
import numpy as np

def _distort(image):

    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        np.clip(tmp, 0, 255, out=tmp)
        image[:] = tmp

    image = image.copy()

    if random.randrange(2):
        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))

        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

        if random.randrange(2):
            _convert(image[:, :, 0], beta=random.uniform(-18, 18))

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    else:
        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

        if random.randrange(2):
            _convert(image[:, :, 0], beta=random.uniform(-18, 18))

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))

    return image

def _mirror(image, boxes):
    h, w, c = image.shape
    if random.randrange(2):
        image = image[:, ::-1, :].copy()
        boxes = boxes.copy()
        boxes[:, 0::2] = w - boxes[:, 0::2]
    return image, boxes

def _resize_subtract_mean(image, size=(1280, 800), rgb_mean=(104, 117, 123)):
    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    interp_method = interp_methods[random.randrange(5)]
    image = cv2.resize(image, size, interpolation=interp_method)
    image = image.astype(np.float32)
    image -= rgb_mean
    return image.transpose(2, 0, 1)

class PreProcess(object):
    def __init__(self, size=(1280, 800), rgb_means=(104, 117, 123)):
        self.size = size
        self.rgb_means = rgb_means

    def __call__(self, image, targets):
        assert targets.shape[0] > 0, 'this image does not have groundtruth'
        boxes = targets[:, :-1].copy()
        labels = targets[:, -1].copy()

        image_t = _distort(image)
        image_t, boxes_t = _mirror(image_t, boxes)
        height, width, _ = image_t.shape
        image_t = _resize_subtract_mean(image_t, size=self.size, rgb_mean=self.rgb_means)

        boxes_t[:, 0::2] *= self.size[0] / width
        boxes_t[:, 1::2] *= self.size[1] / height

        labels_t = np.expand_dims(labels, 1)
        targets_t = np.hstack((boxes_t, labels_t))

        return image_t, targets_t