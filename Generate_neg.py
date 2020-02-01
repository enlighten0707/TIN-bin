import pickle
import numpy as np
import detectron2
import torch
import json
from tqdm import tqdm


def restruct_pose(train_pose):
    pose = {}
    for item in train_pose:
        key = item['image_id']
        if key not in pose:
            pose[key] = []
        pose[key].append([item['bboxes'], item['keypoints']])
    return pose

def map_17_to_16(joint_17):
    joint_16 = np.zeros((16, 3), dtype=np.float32)
    dict_map = {0:16, 1:14, 2:12, 3:11, 4:13, 5:15, 6:[11, 12], 7:[5,6], 9:[1,2], 10:10, 11:8, 12:6, 13:5, 14:7, 15:9}
    for idx in range(16):
        if idx == 8:
            continue # deal thrx joint later
        elif idx in [6, 7, 9]:
            #calc Pelv joint from two hip joint
            #calc neck joint from two shoulder joint
            #calc head joint from two eye joint
            joint_16[idx, 0] = (joint_17[dict_map[idx][0], 0] + joint_17[dict_map[idx][1], 0]) * 0.5
            joint_16[idx, 1] = (joint_17[dict_map[idx][0], 1] + joint_17[dict_map[idx][1], 1]) * 0.5
            joint_16[idx, 2] = (joint_17[dict_map[idx][0], 2] + joint_17[dict_map[idx][1], 2]) * 0.5
        else:
            joint_16[idx] = joint_17[dict_map[idx]]
    #calc thrx joint from head joint and neck, assume the distance is 1:3
    joint_16[8, 0] = joint_16[7, 0] * 0.75  + joint_16[9, 0] * 0.25
    joint_16[8, 1] = joint_16[7, 1] * 0.75  + joint_16[9, 1] * 0.25
    joint_16[8, 2] = joint_16[7, 2] * 0.75  + joint_16[9, 2] * 0.25

    return joint_16

def output_part_box(joint, img_bbox):
    
    flag_bad_joint = 0

    # 16 part names correspond to the center of 16 input joint
    part_size = [1.2, 1, 1, 1.2, 1.2, 1.2, 0.9, 1, 1, 0.9]
    part = [0, 1, 4, 5, 6, 9, 10, 12, 13, 15]

    height = get_distance(joint, 6, 8)

    if (joint[8, 2] < 0.2) or (joint[6, 2] < 0.2):
        flag_bad_joint = 1

    group_score_head      = (joint[7, 2] + joint[8, 2] + joint[9, 2]) / 3
    group_score_left_arm  = (joint[13, 2] + joint[14, 2] + joint[15, 2]) / 3
    group_score_right_arm = (joint[10, 2] + joint[11, 2] + joint[12, 2]) / 3
    group_score_left_leg  = (joint[3, 2] + joint[4, 2] + joint[5, 2]) / 3
    group_score_right_leg = (joint[0, 2] + joint[1, 2] + joint[2, 2]) / 3
    
    # 'Pelv'&'Neck' scaling by the distance of Pelv and Neck
    bbox = np.zeros((10, 4), dtype=np.float32)
    for i in range(10):
        score_joint = joint[part[i], 2]

        # the keypoint is not reliable/ cannot be seen / do not exist
        if (score_joint < 0.2):
            bbox[i, 0] = img_bbox[0]
            bbox[i, 1] = img_bbox[1]
            bbox[i, 2] = img_bbox[2]
            bbox[i, 3] = img_bbox[3]

        # the keypoint is reliable, but the distance cannot be measured by distance between pelv and neck
        elif (score_joint >= 0.2) and (flag_bad_joint == 1):
            if i == 5: # head group
                if group_score_head > 0.2:
                    half_box_width = get_distance(joint, 7, 9)
                    pbox = get_part_box(i, joint, half_box_width)
                    bbox[i, 0] = pbox[0]
                    bbox[i, 1] = pbox[1]
                    bbox[i, 2] = pbox[2]
                    bbox[i, 3] = pbox[3]
                else:
                    bbox[i, 0] = img_bbox[0]
                    bbox[i, 1] = img_bbox[1]
                    bbox[i, 2] = img_bbox[2]
                    bbox[i, 3] = img_bbox[3]
            elif i in [6,7]: # right arm group
                if group_score_right_arm > 0.2: 
                    half_box_width = get_distance(joint, 10, 12)
                    pbox = get_part_box(i, joint, half_box_width)
                    bbox[i, 0] = pbox[0]
                    bbox[i, 1] = pbox[1]
                    bbox[i, 2] = pbox[2]
                    bbox[i, 3] = pbox[3]
                else:
                    bbox[i, 0] = img_bbox[0]
                    bbox[i, 1] = img_bbox[1]
                    bbox[i, 2] = img_bbox[2]
                    bbox[i, 3] = img_bbox[3]
            elif i in [9,10]: # left arm group
                if group_score_left_arm > 0.2: 
                    half_box_width = get_distance(joint, 13, 15)
                    pbox = get_part_box(i, joint, half_box_width)
                    bbox[i, 0] = pbox[0]
                    bbox[i, 1] = pbox[1]
                    bbox[i, 2] = pbox[2]
                    bbox[i, 3] = pbox[3]
                else:
                    bbox[i, 0] = img_bbox[0]
                    bbox[i, 1] = img_bbox[1]
                    bbox[i, 2] = img_bbox[2]
                    bbox[i, 3] = img_bbox[3]
            elif i in [0,1]: # right leg group
                if group_score_right_leg > 0.2: 
                    half_box_width = get_distance(joint, 0, 2)
                    pbox = get_part_box(i, joint, half_box_width)
                    bbox[i, 0] = pbox[0]
                    bbox[i, 1] = pbox[1]
                    bbox[i, 2] = pbox[2]
                    bbox[i, 3] = pbox[3]
                else:
                    bbox[i, 0] = img_bbox[0]
                    bbox[i, 1] = img_bbox[1]
                    bbox[i, 2] = img_bbox[2]
                    bbox[i, 3] = img_bbox[3]
            elif i in [2,3]: # right arm group
                if group_score_left_leg > 0.2: 
                    half_box_width = get_distance(joint, 3, 5)
                    pbox = get_part_box(i, joint, half_box_width)
                    bbox[i, 0] = pbox[0]
                    bbox[i, 1] = pbox[1]
                    bbox[i, 2] = pbox[2]
                    bbox[i, 3] = pbox[3]
                else:
                    bbox[i, 0] = img_bbox[0]
                    bbox[i, 1] = img_bbox[1]
                    bbox[i, 2] = img_bbox[2]
                    bbox[i, 3] = img_bbox[3]
            else: # pelv keypoint
                bbox[i, 0] = img_bbox[0]
                bbox[i, 1] = img_bbox[1]
                bbox[i, 2] = img_bbox[2]
                bbox[i, 3] = img_bbox[3]
    
        else: # the keypoint is reliable and the distance can be measured by distance between pelv and neck
            half_box_width = height * part_size[i] / 2
            pbox = get_part_box(i, joint, half_box_width)
            bbox[i, 0] = pbox[0]
            bbox[i, 1] = pbox[1]
            bbox[i, 2] = pbox[2]
            bbox[i, 3] = pbox[3]
    return np.concatenate([np.zeros((10, 1)), bbox], axis=1)

def get_part_box(i, joint, half_box_width):
    part = [0, 1, 4, 5, 6, 9, 10, 12, 13, 15]
    center_x = joint[part[i], 0]
    center_y = joint[part[i], 1]
    return center_x - half_box_width, center_y - half_box_width, center_x + half_box_width, center_y + half_box_width

def get_distance(joint, keypoint1, keypoint2):
    height_y = joint[keypoint1, 1] - joint[keypoint2, 1] 
    height_x = joint[keypoint1, 0] - joint[keypoint2, 0]
    return np.sqrt(height_x ** 2 + height_y ** 2)

def check_iou(human_bbox_pkl, human_bbox_pose):

    x1, y1, x2, y2 = human_bbox_pose
    x1d, y1d, x2d, y2d = human_bbox_pkl

    xa = max(x1, x1d)
    ya = max(y1, y1d)
    xb = min(x2, x2d)
    yb = min(y2, y2d)

    iw1 = xb - xa + 1
    iw2 = yb - ya + 1

    if iw1 > 0 and iw2 > 0:
        inter_area = iw1 * iw2
        a_area = (x2 - x1) * (y2 - y1)
        b_area = (x2d - x1d) * (y2d - y1d)
        union_area = a_area + b_area - inter_area
        return inter_area / float(union_area)
    else:
        return 0

image_folder_list = {'hico-train': '/Disk1/yonglu/iCAN/Data/hico_20160224_det/images/train2015/', 
                     'hico-test': '/Disk1/yonglu/iCAN/Data/hico_20160224_det/images/test2015/',
                     'hcvrd': '/Disk2/yonglu/behaviour_universe/hcvrd/image/', 
                     'openimage': '/Disk2/yonglu/behaviour_universe/openimage/image/',
                     'vcoco': '/Disk1/yonglu/iCAN/Data/v-coco/coco/images/all/', 
                     'pic': '/Disk2/yonglu/behaviour_universe/pic/All/', 
                     'long_tail_1': '/Disk2/yonglu/behaviour_universe/long_tail_round_1/long_tail_final/img_all/all/', 
                     'long_tail_2': '/Disk2/yonglu/behaviour_universe/long_tail_round_2/img_all/'}
pose_list = {
    'long_tail_2': restruct_pose(json.load(open('/Disk2/yonglu/behaviour_universe/long_tail_round_2/pose_results/alphapose-results.json'))),
    'long_tail_1': restruct_pose(json.load(open('/Disk2/yonglu/behaviour_universe/long_tail_round_1/long_tail_final/alphapose/alphapose-results.json'))),
    'pic':         restruct_pose(json.load(open('/Disk2/yonglu/behaviour_universe/pic/alphapose/alphapose-results.json'))),
    'openimage':   restruct_pose(json.load(open('/Disk2/yonglu/behaviour_universe/openimage/alphapose/alphapose-results.json'))),
    'hcvrd':       restruct_pose(json.load(open('/Disk2/yonglu/behaviour_universe/hcvrd/alphapose/alphapose-results.json'))), 
    'hico-train':  restruct_pose(json.load(open('/Disk2/yonglu/behaviour_universe/hico/alphapose_new/nms-0.6/train2015_pose_results.json'))),
    'vcoco':       restruct_pose(json.load(open('/Disk2/yonglu/behaviour_universe/vcoco/alphapose/alphapose-results.json')))
}

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

verb_range = {}
for value in range(len(obj_range)):
    for key in range(obj_range[value][0]-1, obj_range[value][1]):
        verb_range[key] = value + 1

GT    = pickle.load(open('Train_GT_10w.pkl', 'rb'), encoding='latin1')
hico  = pickle.load(open('/Disk1/yonglu/iCAN/Data/Trainval_Neg_HICO_with_pose_with_part_box_iou+sub_addpvp_py2.pkl', 'rb'), encoding='latin1')
vcoco = pickle.load(open('/Disk1/yonglu/iCAN/Data/Trainval_Neg_VCOCO_with_partbox.pkl', 'rb'), encoding='latin1')
Neg, GT_all, missed = {}, {}, []

for key, value in tqdm(GT.items()):
    dataset = value[0][4]
    if dataset == 'hico-test' or 'COCO_val' in key:
        continue
    pose    = pose_list[dataset]
    if key not in pose:
        pose[key] = []
        missed.append(key)
        print(key, dataset, len(missed))
    GT_all[key] = []
    for item in value:
        GT_all[key].append([
            item[0], # path
            item[1], # hoi
            item[2], # hbox
            item[3], # obox
            item[4], # dataset
            verb_range[item[1][0]], # object
            1.0])    # score
        k, miou, hbox = -1, 0.0, item[2]
        for idx in range(len(pose[key])):
            iou = check_iou(hbox, pose[key][idx][0])
            if iou > miou:
                miou = iou
                k    = idx
        if k != -1 and miou > 0.7:
            GT_all[key][-1].append(pose[key][k][1])
            joint_17 = np.array(pose[key][k][1]).reshape(17, 3)
            joint_16 = map_17_to_16(joint_17)
            GT_all[key][-1].append(output_part_box(joint_16, hbox))
        else:
            GT_all[key][-1].append(None)
            GT_all[key][-1].append(None)
        GT_all[key][-1].append(item[-1])
    
    if dataset == 'hico-train':
        if int(key[-9:-4]) in hico.keys():
            Neg[key] = []
            for item in hico[int(key[-9:-4])]:
                Neg[key].append([
                    item[0], # path
                    item[1], # hoi
                    item[2], # hbox
                    item[3], # obox
                    item[5], # obj
                    item[6]]) # score
                k, miou, hbox = -1, 0.0, item[2]
                for idx in range(len(pose[key])):
                    iou = check_iou(hbox, pose[key][idx][0])
                    if iou > miou:
                        miou = iou
                        k    = idx
                if k != -1 and miou > 0.7:
                    Neg[key][-1].append(pose[key][k][1])
                    joint_17 = np.array(pose[key][k][1]).reshape(17, 3)
                    joint_16 = map_17_to_16(joint_17)
                    Neg[key][-1].append(output_part_box(joint_16, hbox))
                else:
                    Neg[key][-1].append(None)
                    Neg[key][-1].append(None)
    elif dataset == 'vcoco':
        if int(key[-16:-4]) in vcoco.keys():
            Neg[key] = []
            for item in vcoco[int(key[-16:-4])]:
                Neg[key].append([
                    item[0],
                    obj_range[item[5]-1][1]-1, 
                    item[2],
                    item[3], 
                    item[5], 
                    item[6], 
                ])
                k, miou, hbox = -1, 0.0, item[2]
                for idx in range(len(pose[key])):
                    iou = check_iou(hbox, pose[key][idx][0])
                    if iou > miou:
                        miou = iou
                        k    = idx
                if k != -1 and miou > 0.7:
                    Neg[key][-1].append(pose[key][k][1])
                    joint_17 = np.array(pose[key][k][1]).reshape(17, 3)
                    joint_16 = map_17_to_16(joint_17)
                    Neg[key][-1].append(output_part_box(joint_16, hbox))
                else:
                    Neg[key][-1].append(None)
                    Neg[key][-1].append(None)
    else:
        tmp = []
        det = pickle.load(open(dataset + '/' + key[:-4] + '.pkl', 'rb'))['instances']
        bboxes  = det.pred_boxes.tensor.cpu().numpy()
        classes = det.pred_classes.cpu().numpy()
        scores  = det.pred_classes.cpu().numpy()
        for i in range(bboxes.shape[0]):
            if classes[i] == 0:
                hbox = bboxes[i]
                for j in range(bboxes.shape[0]):
                    if i == j:
                        continue
                    obox = bboxes[j]
                    flag = False
                    for k in range(len(value)):
                        if check_iou(hbox, value[k][2]) > 0.5 and check_iou(obox, value[k][3]) > 0.5:
                            flag = True
                            break
                    if flag:
                        continue
                    else:
                        tmp.append([key, obj_range[classes[j]][1]-1, hbox, obox, classes[j]+1, scores[j]])
                        k, miou = -1, 0.0
                        for idx in range(len(pose[key])):
                            iou = check_iou(hbox, pose[key][idx][0])
                            if iou > miou:
                                miou = iou
                                k = idx
                        if k != -1 and miou > 0.7:
                            tmp[-1].append(pose[key][k][1])
                            joint_17 = np.array(pose[key][k][1]).reshape(17, 3)
                            joint_16 = map_17_to_16(joint_17)
                            tmp[-1].append(output_part_box(joint_16, hbox))
                        else:
                            tmp[-1].append(None)
                            tmp[-1].append(None)
        if len(tmp) > 0:
            Neg[key] = tmp
pickle.dump(GT_all, open('Train_GT_10w_new.pkl', 'wb'), protocol=2)
pickle.dump(Neg, open('Train_Neg_10w_new.pkl', 'wb'), protocol=2)
