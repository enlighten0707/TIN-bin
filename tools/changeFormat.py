import cPickle as pickle
import numpy as np

prior_obj = [
    [38], [31], [43, 44, 77], range(80), 
    # [1, 37, 46, 47, 48, 49, 50, 52, 53, 54, 55, 56, 64, 74], 
    [37, 46, 47, 48, 49, 50, 52, 53, 54, 55, 56, 64, 74], 
    [2, 3, 4, 6, 7, 8, 9, 18, 21], 
    [68], [33], [64], 
    [47, 48, 49, 50, 51, 52, 53, 54, 55, 56], 
    [2, 4, 14, 18, 21, 29, 57, 58, 60, 61, 62], 
    [31, 32, 37, 38], 
    [14, 57, 58, 60, 62], 
    [40, 41, 42, 46], 
    # [1, 2, 3, 17, 19, 25, 26, 27, 29, 30, 31, 32, 34, 35, 37, 38, 47, 55, 64, 68, 78], 
    [2, 3, 17, 19, 25, 26, 27, 29, 30, 31, 32, 34, 35, 37, 38, 47, 55, 64, 68, 78], 
    [30, 33], [43, 44, 45], range(80), 
    # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 53, 54, 55, 56, 57, 59, 61, 63, 64, 65, 66, 67, 68, 74, 75, 77], 
    [2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 53, 54, 55, 56, 57, 59, 61, 63, 64, 65, 66, 67, 68, 74, 75, 77], 
    [35, 39], [33], [32], range(80), 
    # [1, 4, 29, 31, 33, 34, 47, 49, 53, 54, 60, 63, 64, 73], 
    [4, 29, 31, 33, 34, 47, 49, 53, 54, 60, 63, 64, 73], 
    [74], 
    # [1, 2, 4, 8, 9, 14, 16, 17, 18, 19, 20, 21, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 53, 54, 55, 56, 57, 60, 61, 64, 65, 66, 67, 68, 74, 77, 78, 79, 80], 
    [2, 4, 8, 9, 14, 16, 17, 18, 19, 20, 21, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 53, 54, 55, 56, 57, 60, 61, 64, 65, 66, 67, 68, 74, 77, 78, 79, 80], 
    [37], range(80), 
    [30, 33]
]

iCAN_to_vcoco = {
    25: (0, 0), 27: (1, 0), 10: (2, 0), 5: (3, 0), 3: (4, 0), 18: (5, 0), 19: (6, 0), 
    20: (6, 1), 9: (7, 0), 16: (7, 1), 11: (8, 0), 12: (9, 0), 6: (10, 0), 14: (11, 0), 
    15: (12, 0), 28: (13, 0), 2: (14, 0), 4: (14, 1), 22: (15, 0), 8: (16, 0), 
    1: (17, 0), 0: (18, 0), 26: (19, 0), 17: (20, 0), 13: (21, 0), 7: (22, 0), 
    23: (23, 0), 24: (24, 0), 21: (25, 0)
}

s = pickle.load(open('S.pkl', 'rb'))
d = pickle.load(open('D.pkl', 'rb'))

data = {}
prior_mask = pickle.load( open( '/Disk1/yonglu/iCAN/Data/prior_mask.pkl', "rb" ) )
for tr in range(len(s)):
    dic_s        = s[tr]
    binary_score = d[tr]
    key          = dic_s['image_id']
    if key not in data:
        data[key] = []
    this_agent = np.concatenate([dic_s['person_box'], dic_s['H_Score']])
    this_agent = np.append(this_agent, dic_s['H_det']).reshape(1, -1)
    this_role  = []
    for i in range(dic_s['HO_Score'].shape[0]):
        HO_Score = dic_s['HO_Score'][i] * prior_mask[:, dic_s['object_class'][i]].reshape(-1)
        for j in range(29):
            if dic_s['object_class'][i] not in prior_obj[j]:
                HO_Score[j] = np.nan
        this_obj = np.concatenate([dic_s['object_box'][i], HO_Score, binary_score[i]]).reshape(1, -1)
        this_obj = np.append(this_obj, dic_s['object_class'][i])
        this_obj = np.append(this_obj, dic_s['O_det'][i]).reshape(1, -1)
        this_role.append(this_obj)
    this_role = np.concatenate(this_role, axis=0)
    # print(this_role.shape)
    # print(this_agent.shape)
    data[key].append([this_agent, this_role])
    if tr % 100 == 0:
        print(tr)

pickle.dump(data, open('vcoco_best_new_format.pkl', 'wb'))
    