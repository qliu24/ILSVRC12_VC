import numpy as np
import pickle
import sys
from config_voting_ILSVRC12 import *

# subset_idx = int(sys.argv[1])
# candidates_cls = np.where(subset_lb==subset_idx)[0]
# for cls_idx in candidates_cls:
#     fname = '/export/home/qliu24/ILSVRC12_VC/result/scores_obj{}.pickle'.format(cls_idx)

#     order_idx = np.where(candidates_cls==cls_idx)[0][0]

#     with open(fname, 'rb') as fh:
#         scores = pickle.load(fh)

#     rst=np.argmax(scores, axis=1)
#     max_rst = np.argsort(-np.bincount(rst))
#     for ii in max_rst[0:3]:
#         print('class {}: {} samples were classified as class {}'.format(cls_idx, np.sum(rst==ii), candidates_cls[ii]))
        
        
cls_idx = int(sys.argv[1])
subset_idx = subset_lb[cls_idx]
candidates_cls = np.where(subset_lb==subset_idx)[0]
order_idx = np.where(candidates_cls==cls_idx)[0][0]
# fname = '/export/home/qliu24/ILSVRC12_VC/result/scores_all_obj{}.pickle'.format(cls_idx)
fname = '/export/home/qliu24/ILSVRC12_VC/result/scores_obj{}.pickle'.format(cls_idx)
with open(fname, 'rb') as fh:
    # _,scores = pickle.load(fh)
    scores = pickle.load(fh)

rst=np.argmax(scores, axis=1)
max_rst = np.argsort(-np.bincount(rst))
# for ii in max_rst[0:3]:
#     print('class {}: {} samples were classified as class {}'.format(cls_idx, np.sum(rst==ii), ii))

# print('class {}, {}, ({})'.format(cls_idx, np.sum(rst==cls_idx)/len(rst), max_rst[0:3]))
print('class {}, {}'.format(cls_idx, np.sum(rst==order_idx)/len(rst)))


