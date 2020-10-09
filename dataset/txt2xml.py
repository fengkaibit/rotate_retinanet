import cv2
import os
import random
import math

headstr = """\
<annotation>
    <folder>UCAS_AOD</folder>
    <filename>%s</filename>
    <source>
        <database>My Database</database>
        <annotation>VOC</annotation>
        <image>flickr</image>
        <flickrid>NULL</flickrid>
    </source>
    <size>
        <width>%d</width>
        <height>%d</height>
        <depth>%d</depth>
    </size>
    <segmented>0</segmented>
"""
objstr = """\
    <object>
        <name>%s</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <x1>%s</x1>
            <y1>%s</y1>
            <x2>%s</x2>
            <y2>%s</y2>
            <x3>%s</x3>
            <y3>%s</y3>
            <x4>%s</x4>
            <y4>%s</y4>
        </bndbox>
    </object>
"""
tailstr = '''\
</annotation>
'''

def load_txt(txt_path):
    label = txt_path.split('/')[-1].split('.')[0].split('_')[0]
    boxes, labels = [], []
    with open(txt_path, 'r') as f:
        line = f.readlines()
        for l in line:
            b = l.split('\n')[0].split('\t')[:8]
            b = list(map(float, b))
            boxes.append(b)
            labels.append(label)

    return boxes, labels

def write_xml(boxes, labels, xml_path, w, h, c):
    img_name = xml_path.split('/')[-1].split('.')[0]
    f = open(xml_path, "w")
    head = headstr % (img_name, w, h, c)
    f.write(head)
    for name, obj in zip(labels, boxes):
        f.write(objstr % (name, obj[0], obj[1], obj[2], obj[3], obj[4], obj[5], obj[6], obj[7]))
    tail = tailstr
    f.write(tail)

def create_xml_annotations(image_folder_path, txt_folder_path, xml_folder_path):
    for txt_name in os.listdir(txt_folder_path):
        txt_path = os.path.join(txt_folder_path, txt_name)
        boxes, labels = load_txt(txt_path)
        img_path = os.path.join(image_folder_path, txt_name.split('.')[0] + '.png')
        img = cv2.imread(img_path)
        h, w, c = img.shape
        xml_path = os.path.join(xml_folder_path, txt_name.split('.')[0] + '.xml')
        write_xml(boxes, labels, xml_path, w, h, c)

def split_dataset(image_folder_path, txt_save_path, split_ratio=1.0):
    imgs = os.listdir(image_folder_path)
    random.shuffle(imgs)
    train_imgs = imgs[:int(math.ceil(len(imgs)) * split_ratio)]
    test_imgs = imgs[int(math.ceil(len(imgs)) * split_ratio):]
    train_txt_path = txt_save_path + 'train_list.txt'
    test_txt_path = txt_save_path + 'test_list.txt'
    with open(train_txt_path, "w") as f:
        for train_img in train_imgs:
            f.writelines(train_img + '\n')
    with open(test_txt_path, "w") as f:
        for test_img in test_imgs:
            f.writelines(test_img + '\n')


if __name__ == '__main__':
    image_folder_path = '/home/fengkai/datasets/UCAS-AOD/JPEGImages/'
    txt_folder_path = '/home/fengkai/datasets/UCAS-AOD/txt_annotations/'
    xml_folder_path = '/home/fengkai/datasets/UCAS-AOD/Annotations/'
    txt_save_path = '/home/fengkai/datasets/UCAS-AOD/'
    #create_xml_annotations(image_folder_path, txt_folder_path, xml_folder_path)
    #split_dataset(image_folder_path, txt_save_path, split_ratio=0.75)