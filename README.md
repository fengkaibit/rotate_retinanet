# rotate_retinanet
rotate retinanet

# dataset
数据集使用UCAS-AOD数据集，包含plane和car两个类别的旋转框标注。
将plane和car两个类别的图片和txt标注按照voc数据集格式添加到JPEGImages和Annotations文件夹中。
使用txt2xml.py生成voc格式的xml标注。

# 编译iou和nms
cd utils/cython_utils

 python3 setup.py build_ext --inplace

cd utils/rotate_cython_utils

 python3 setup.py build_ext --inplace
 
 # train
 python3 train.py
 
 # eval
 python3 eval/voc_eval_r.py
 
 # demo
 python3 demo.py
