
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cPickle as pickle
import numpy as np
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

obj_range = [
    (161, 170), (11, 24), (66, 76), (147, 160), (1, 10), 
    (55, 65), (187, 194), (568, 576), (32, 46), (563, 567), 
    (326, 330), (503, 506), (415, 418), (244, 247), (25, 31), 
    (77, 86), (112, 129), (130, 146), (175, 186), (97, 107), 
    (314, 325), (236, 239), (596, 600), (343, 348), (209, 214), 
    (577, 584), (353, 356), (539, 546), (507, 516), (337, 342), 
    (464, 474), (475, 483), (489, 502), (369, 376), (225, 232), 
    (233, 235), (454, 463), (517, 528), (534, 538), (47, 54), 
    (589, 595), (296, 305), (331, 336), (377, 383), (484, 488), 
    (253, 257), (215, 224), (199, 208), (439, 445), (398, 407), 
    (258, 264), (274, 283), (357, 363), (419, 429), (306, 313), 
    (265, 273), (87, 92), (93, 96), (171, 174), (240, 243), 
    (108, 111), (551, 558), (195, 198), (384, 389), (394, 397), 
    (435, 438), (364, 368), (284, 290), (390, 393), (408, 414), 
    (547, 550), (450, 453), (430, 434), (248, 252), (291, 295), 
    (585, 588), (446, 449), (529, 533), (349, 352), (559, 562)
]

obj_name = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 
    'bus', 'train', 'truck', 'boat', 'traffic_light',
    'fire_hydrant', 'stop_sign', 'parking_meter', 'bench', 'bird',
    'cat', 'dog', 'horse', 'sheep', 'cow', 
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
    'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket', 'bottle',
    'wine_glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'potted_plant', 'bed',
    'dining_table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell_phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush',
]

def iou(bb1, bb2):
    x1 = bb1[2] - bb1[0]
    y1 = bb1[3] - bb1[1]
    if x1 > 0 and y1 > 0:
        s1 = x1 * y1
    else:
        s1 = 0

    x2 = bb2[1] - bb2[0]
    y2 = bb2[3] - bb2[2]
    if x2 > 0 and y2 > 0:
        s2 = x2 * y2
    else:
        s2 = 0

    xiou = min(bb1[2], bb2[1]) - max(bb1[0], bb2[0])
    yiou = min(bb1[3], bb2[3]) - max(bb1[1], bb2[2])
    if xiou > 0 and yiou > 0:
        siou = xiou * yiou
    else:
        siou = 0
    
    bottom = s1 + s2 - siou
    if bottom <= 0:
        return 0
    else:
        return siou / bottom

def calc_hit(det, gtbox):
    gtbox = gtbox.astype(np.float64)
    hiou = iou(det[:4], gtbox[:4])
    oiou = iou(det[4:], gtbox[4:])
    return min(hiou, oiou)

def coco_map(gt_bbox, idx, bboxes, keys, x, y, npos):
    hit, rec = [], []
    used = set()
    for i in range(len(idx)):
        pair_id = idx[i]
        bbox    = bboxes[pair_id, :]
        key     = keys[pair_id]
        hit.append(0)
        rec.append(0)
        for action in range(x, y):
            if key in gt_bbox[action]:
                maxi, k = 0.0, -1
                for j in range(gt_bbox[action][key].shape[0]):
                    tmp = calc_hit(bbox, gt_bbox[action][key][j, :])
                    if maxi < tmp:
                        maxi = tmp
                        k    = j
                if maxi > 0.5:
                    hit[-1] = 1
                    if (action, key, k) not in used:
                        used.add((action, key, k))
                        rec[-1] += 1
    bottom = np.array(range(len(hit))) + 1
    hit    = np.cumsum(hit)
    prec   = hit / bottom
    rec    = np.cumsum(rec) / npos
    ap     = 0.0
    for i in range(11):
        mask = rec >= (i / 10.0)
        if np.sum(mask) > 0:
            ap += np.max(prec[mask]) / 11.0
    return ap, prec, rec

def calc_map(scores, bboxes, keys, classid, nis_dir):
    gt_bbox = {}
    x = obj_range[classid][0] - 1
    y = obj_range[classid][1] - 1
    npos = 0
    for i in range(x, y):
        anno = pickle.load(open('gt_hoi_py2/hoi_%d.pkl' % i, 'rb'))
        gt_bbox[i] = anno
        for key in anno.keys():
            npos += anno[key].shape[0]
    
    res_all = []
    
    idx = np.argsort(scores[:, 0])[::-1]
    ap, prec, rec = coco_map(gt_bbox, idx, bboxes, keys, x, y, npos)
    fig = plt.figure()
    plt.plot(prec, color='b')
    plt.plot(rec,  color='r')
    if not os.path.exists(nis_dir + '/pos/'):
        os.mkdir(nis_dir + '/pos/')
    fig.savefig(nis_dir + '/pos/%s.jpg' % obj_name[classid])
    plt.close()
    res_all.append((ap, np.max(rec)))

    idx = np.argsort(scores[:, 1])
    ap, prec, rec = coco_map(gt_bbox, idx, bboxes, keys, x, y, npos)
    fig = plt.figure()
    plt.plot(prec, color='b')
    plt.plot(rec,  color='r')
    if not os.path.exists(nis_dir + '/neg/'):
        os.mkdir(nis_dir + '/neg/')
    fig.savefig(nis_dir + '/neg/%s.jpg' % obj_name[classid])
    plt.close()
    res_all.append((ap, np.max(rec)))

    idx = np.argsort(scores[:, 0] - scores[:, 1])[::-1]
    ap, prec, rec = coco_map(gt_bbox, idx, bboxes, keys, x, y, npos)
    fig = plt.figure()
    plt.plot(prec, color='b')
    plt.plot(rec,  color='r')
    if not os.path.exists(nis_dir + '/pos-neg/'):
        os.mkdir(nis_dir + '/pos-neg/')
    fig.savefig(nis_dir + '/pos-neg/%s.jpg' % obj_name[classid])
    plt.close()
    res_all.append((ap, np.max(rec)))

    return res_all
    
def Generate_NIS_map(bboxes, binary, keys, nis_dir):
    
    map, mrec   = np.zeros((3, 80)), np.zeros((3, 80))
    for i in range(80):
#        bbox    = np.concatenate(bboxes[i], axis=0)
#        nis_all = np.concatenate(binary[i], axis=0)
#        if keys[i] is not None:
#            key     = np.concatenate(keys[i], axis=0)
        bbox    = np.array(bboxes[i])
        nis_all = np.array(binary[i])
        key     = np.array(keys[i])
        res     = calc_map(nis_all, bbox, key, i, nis_dir)
        for j in range(3):
           map[j, i], mrec[j, i] = res[j]
        print(obj_name[i])
    return map, mrec