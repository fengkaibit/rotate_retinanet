import xml.etree.ElementTree as ET
import os
from utils.iou import rotate_bbox_iou
from dataset.voc_dataset import NAME_LABEL_MAP
from utils.box_utils import points8_to_center, center_to_points8
import torch
import cv2
import random
import numpy as np
from models.retinanet import build_retinanet
import argparse

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def write_voc_results_file(all_boxes, test_imgid_list, det_save_dir):
  '''

  :param all_boxes: is a list. each item reprensent the detections of a img.
  the detections is a array. shape is [-1, 7]. [category, score, x, y, w, h, theta]
  Note that: if none detections in this img. that the detetions is : []

  :param test_imgid_list:
  :param det_save_path:
  :return:
  '''
  for cls, cls_id in NAME_LABEL_MAP.items():
    if cls == 'back_ground':
      continue
    print("Writing {} VOC resutls file".format(cls))

    mkdir(det_save_dir)
    det_save_path = os.path.join(det_save_dir, "det_"+cls+".txt")
    with open(det_save_path, 'wt') as f:
      for index, img_name in enumerate(test_imgid_list):
        this_img_detections = all_boxes[index]
        this_img_detections = np.array(this_img_detections)
        this_cls_detections = this_img_detections[this_img_detections[:, 0] == cls_id]
        if this_cls_detections.shape[0] == 0:
          continue # this cls has none detections in this img
        for a_det in this_cls_detections:
          f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                  format(img_name, a_det[1],
                         a_det[2], a_det[3],
                         a_det[4], a_det[5], a_det[6]))  # that is [img_name, score, x, y, w, h, theta]


def parse_rec(filename):
  """ Parse a PASCAL VOC xml file """
  tree = ET.parse(filename)
  objects = []
  for obj in tree.findall('object'):
    obj_struct = {}
    obj_struct['name'] = obj.find('name').text
    obj_struct['pose'] = obj.find('pose').text
    obj_struct['truncated'] = int(obj.find('truncated').text)
    obj_struct['difficult'] = int(obj.find('difficult').text)
    bbox = obj.find('bndbox')
    rbox = [int(float(bbox.find('x1').text)), int(float(bbox.find('y1').text)),
            int(float(bbox.find('x2').text)), int(float(bbox.find('y2').text)),
            int(float(bbox.find('x3').text)), int(float(bbox.find('y3').text)),
            int(float(bbox.find('x4').text)), int(float(bbox.find('y4').text))]
    rbox = np.array([rbox], np.float32)
    rbox = points8_to_center(rbox)
    obj_struct['bbox'] = rbox
    objects.append(obj_struct)

  return objects


def voc_ap(rec, prec, use_07_metric=False):
  """ ap = voc_ap(rec, prec, [use_07_metric])
  Compute VOC AP given precision and recall.
  If use_07_metric is true, uses the
  VOC 07 11 point method (default:False).
  """
  if use_07_metric:
    # 11 point metric
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
      if np.sum(rec >= t) == 0:
        p = 0
      else:
        p = np.max(prec[rec >= t])
      ap = ap + p / 11.
  else:
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
      mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
  return ap


def voc_eval(detpath, annopath, test_imgid_list, cls_name, ovthresh=0.5,
             use_07_metric=False, use_diff=False):
  '''

  :param detpath:
  :param annopath:
  :param test_imgid_list: it 's a list that contains the img_name of test_imgs
  :param cls_name:
  :param ovthresh:
  :param use_07_metric:
  :param use_diff:
  :return:
  '''
  # 1. parse xml to get gtboxes

  # read list of images
  imagenames = test_imgid_list

  recs = {}
  for i, imagename in enumerate(imagenames):
    recs[imagename] = parse_rec(os.path.join(annopath, imagename+'.xml'))
    # if i % 100 == 0:
    #   print('Reading annotation for {:d}/{:d}'.format(
    #     i + 1, len(imagenames)))

  # 2. get gtboxes for this class.
  class_recs = {}
  num_pos = 0
  # if cls_name == 'person':
  #   print ("aaa")
  for imagename in imagenames:
    R = [obj for obj in recs[imagename] if obj['name'] == cls_name]
    bbox = np.array([x['bbox'] for x in R])
    if use_diff:
      difficult = np.array([False for x in R]).astype(np.bool)
    else:
      difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
    det = [False] * len(R)
    num_pos = num_pos + sum(~difficult)  # ignored the diffcult boxes
    class_recs[imagename] = {'bbox': bbox,
                             'difficult': difficult,
                             'det': det} # det means that gtboxes has already been detected

  # 3. read the detection file
  detfile = os.path.join(detpath, "det_"+cls_name+".txt")
  with open(detfile, 'r') as f:
    lines = f.readlines()

  # for a line. that is [img_name, confidence, xmin, ymin, xmax, ymax]
  splitlines = [x.strip().split(' ') for x in lines]  # a list that include a list
  image_ids = [x[0] for x in splitlines]  # img_id is img_name
  confidence = np.array([float(x[1]) for x in splitlines])
  BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

  nd = len(image_ids) # num of detections. That, a line is a det_box.
  tp = np.zeros(nd)
  fp = np.zeros(nd)

  if BB.shape[0] > 0:
    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]  #reorder the img_name

    # go down dets and mark TPs and FPs
    for d in range(nd):
      R = class_recs[image_ids[d]]  # img_id is img_name
      bb = BB[d, :].astype(float)
      ovmax = -np.inf
      BBGT = R['bbox'].astype(float)

      if BBGT.size > 0:
        # compute overlaps
        # intersection
        # ixmin = np.maximum(BBGT[:, 0], bb[0])
        # iymin = np.maximum(BBGT[:, 1], bb[1])
        # ixmax = np.minimum(BBGT[:, 2], bb[2])
        # iymax = np.minimum(BBGT[:, 3], bb[3])
        # iw = np.maximum(ixmax - ixmin + 1., 0.)
        # ih = np.maximum(iymax - iymin + 1., 0.)
        # inters = iw * ih
        #
        # # union
        # uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
        #        (BBGT[:, 2] - BBGT[:, 0] + 1.) *
        #        (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)
        #
        # overlaps = inters / uni
        overlaps = []
        for i in range(len(BBGT)):
          overlap = rotate_bbox_iou(np.array([bb], dtype=np.float32),BBGT[i].astype(np.float32))[0]
          overlaps.append(overlap)
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)

      if ovmax > ovthresh:
        if not R['difficult'][jmax]:
          if not R['det'][jmax]:
            tp[d] = 1.
            R['det'][jmax] = 1
          else:
            fp[d] = 1.
      else:
        fp[d] = 1.

  # 4. get recall, precison and AP
  fp = np.cumsum(fp)
  tp = np.cumsum(tp)
  rec = tp / float(num_pos)
  # avoid divide by zero in case the first detection matches a difficult
  # ground truth
  prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
  ap = voc_ap(rec, prec, use_07_metric)

  return rec, prec, ap


def do_python_eval(test_imgid_list, test_annotation_path, det_save_dir, use_07_metric, use_diff):
  # import matplotlib.colors as colors
  # import matplotlib.pyplot as plt

  AP_list = []
  for cls, index in NAME_LABEL_MAP.items():
    if cls == 'back_ground':
      continue
    recall, precision, AP = voc_eval(detpath=det_save_dir,
                                     test_imgid_list=test_imgid_list,
                                     cls_name=cls,
                                     annopath=test_annotation_path,
                                     use_07_metric=use_07_metric,
                                     ovthresh=0.5,
                                     use_diff=use_diff)
    AP_list += [AP]
    print("cls : {}|| Recall: {} || Precison: {}|| AP: {}".format(cls, recall[-1], precision[-1], AP))
    # print("{}_ap: {}".format(cls, AP))
    # print("{}_recall: {}".format(cls, recall[-1]))
    # print("{}_precision: {}".format(cls, precision[-1]))
    r = np.array(recall)
    p = np.array(precision)
    F1 = 2 * r * p / (r + p)
    max_ind = np.argmax(F1)
    print('F1:{} P:{} R:{}'.format(F1[max_ind], p[max_ind], r[max_ind]))

    # c = colors.cnames.keys()
    # c_dark = list(filter(lambda x: x.startswith('dark'), c))
    # c = ['red', 'orange']
    # plt.axis([0, 1.2, 0, 1])
    # plt.plot(recall, precision, color=c_dark[index], label=cls)

  # plt.legend(loc='upper right')
  # plt.xlabel('R')
  # plt.ylabel('P')
  # plt.savefig('./PR_R.png')

  print("mAP is : {}".format(np.mean(AP_list)))


def voc_evaluate_detections(all_boxes, test_imgid_list, test_annotation_path, det_save_dir, use_07_metric, use_diff):
    '''

    :param all_boxes: is a list. each item reprensent the detections of a img.

    The detections is a array. shape is [-1, 6]. [category, score, xmin, ymin, xmax, ymax]
    Note that: if none detections in this img. that the detetions is : []
    :return:
    '''

    write_voc_results_file(all_boxes, test_imgid_list=test_imgid_list,
                         det_save_dir=det_save_dir)
    do_python_eval(test_imgid_list, test_annotation_path, det_save_dir, use_07_metric, use_diff)

def eval(args=None):
    parser = argparse.ArgumentParser(description='model eval.')

    parser.add_argument('--model_path', help='Path to model',
                        default='/home/fengkai/PycharmProjects/my_retinanet_rotate/model_final.pth')
    parser.add_argument('--test_txt_path', help='Path to text_txt',
                        default='/home/fengkai/datasets/UCAS-AOD/test_list.txt')
    parser.add_argument('--images_path', help='Path to JPEGImages Dir',
                        default='/home/fengkai/datasets/UCAS-AOD/JPEGImages')
    parser.add_argument('--annotation_path', help='Path to Annotation Dir',
                        default='/home/fengkai/datasets/UCAS-AOD/Annotations/')
    parser.add_argument('--det_save_dir', help='det save dir',
                        default='./eval_res/')
    parser.add_argument('--use_07_metric', help='use 07 metric', type=bool,
                        default=False)
    parser.add_argument('--use_diff', help='use difficult annotations', type=bool,
                        default=False)

    parser = parser.parse_args(args)

    retinanet = build_retinanet()
    checkpoint = torch.load(parser.model_path)
    retinanet.load_state_dict(checkpoint['state_dict'])

    if torch.cuda.is_available():
        retinanet = retinanet.cuda()

    retinanet.training = False
    retinanet.eval()

    all_boxes = []
    test_imgid_list = []

    with open(parser.test_txt_path) as f:
        line = f.readlines()
        total_nums = len(line)
        print('Num eval images: {}'.format(total_nums))
        for num, l in enumerate(line):
            print(num,'/',total_nums)
            l = l.strip('\n')
            image_name = l.split('.')[0]
            test_imgid_list.append(image_name)
            all_box = []
            image_path = os.path.join(parser.images_path, l)

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
                idxs = np.where(scores.cpu() > 0.05)

                for j in range(idxs[0].shape[0]):
                    bbox = transformed_anchors[idxs[0][j], :]
                    scale_x = size[0] / image_orig.shape[1]
                    scale_y = size[1] / image_orig.shape[0]
                    bbox = torch.unsqueeze(bbox, dim=0)
                    bbox = center_to_points8(bbox)
                    bbox[:, 0::2] /= scale_x
                    bbox[:, 1::2] /= scale_y
                    label_name = int(classification[idxs[0][j]])
                    #score = scores[j]
                    score = scores[idxs[0][j]]

                    bbox = points8_to_center(bbox)[0].tolist()
                    bbox = [label_name, score.cpu().numpy().tolist()] + bbox
                    all_box.append(bbox)
            all_boxes.append(all_box)

    test_annotation_path = parser.annotation_path
    det_save_dir = parser.det_save_dir
    use_07_metric = parser.use_07_metric
    use_diff = parser.use_diff
    voc_evaluate_detections(all_boxes, test_imgid_list, test_annotation_path, det_save_dir, use_07_metric, use_diff)

if __name__ == '__main__':
    eval()


