import cPickle as pickle
import numpy as np
a = pickle.load(open('/Disk1/yonglu/TIN_CVPR_final/-Results/6000_TIN_VCOCO_0.6_0.4_naked.pkl', 'rb'))
b = pickle.load(open('/Disk1/yonglu/TIN_CVPR_final/-Results/60000_TIN_VCOCO_D.pkl', 'rb'))
keys, bbox_H, score_H, det_H = [], [], [], []
ref_O, bbox_O, score_HO, det_O, class_O = [], [], [], [], []

assert len(a) == len(b)

for i in range(len(a)):
    keys.append(a[i]['image_id'])                    # image_id  : scalar
    score_H.append(a[i]['H_Score'])                  # role score: 1*29
    bbox_H.append(a[i]['person_box'].reshape(1, -1)) # person box: 1*4
    tmp_abox = np.array(a[i]['person_box'])
    tmp_bbox = np.array(b[i]['person_box'])
    assert np.all(tmp_abox == tmp_bbox)
    det_H.append(a[i]['H_det'])                      # Human det:  scalar
    if 'object_box' in a[i]:
        assert len(b[i]['object_box']) == len(a[i]['object_box'])
        bbox_O.append(np.array(a[i]['object_box']))             # obj bbox:  num_O * 4
        class_O.append(np.array(a[i]['object_class']))          # obj class: num_O
        det_O.append(np.array(a[i]['O_det']))                   # obj det:   num_O
        score_HO.append(np.squeeze(np.array(a[i]['HO_Score'])).reshape(-1, 29)) # HO score:  num_O * 29
        ref_O.append(np.array([i] * len(a[i]['object_class']))) # reference: num_O

keys    = np.array(keys)
score_H  = np.concatenate(score_H, axis=0)
score_HO = np.concatenate(score_HO, axis=0)
bbox_H  = np.concatenate(bbox_H, axis=0)
bbox_O  = np.concatenate(bbox_O, axis=0)
det_H   = np.array(det_H)
det_O   = np.concatenate(det_O, axis=0)
ref_O   = np.concatenate(ref_O)
class_O = np.concatenate(class_O)

d = {}
d['keys'] = keys
d['score_H'] = np.squeeze(score_H)
d['score_HO'] = score_HO
d['bbox_H'] = bbox_H
d['bbox_O'] = bbox_O
d['det_H']  = det_H
d['det_O']  = det_O
d['ref_O']  = ref_O
d['class_O'] = class_O

print(score_H.shape)
print(score_HO.shape)
pickle.dump(d, open('6000_TIN_VCOCO_0.6_0.4_restruct.pkl', 'wb'))