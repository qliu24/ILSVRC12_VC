from config_voting_ILSVRC12 import *
from vMFMM import *

subset_idx = 7
subset_ls = np.where(subset_lb==subset_idx)[0]

cluster_num = VC['num']
    
file_num = int(math.ceil(len(subset_ls)*200/2000))

# cluster_num = VC['num']
# file_num = 7
feat_set = np.zeros((featDim, 0))
# loc_set = np.zeros((5, 0), dtype='int')
for ii in range(file_num):
    print('loading file {0}'.format(ii))
    fname = Dict['cache_path_sub']+'{}_set{}.pickle'.format(ii, subset_idx)
    # fname = Dict['cache_path']+'{}.pickle'.format(ii)
    with open(fname, 'rb') as fh:
        res, _ = pickle.load(fh)
        feat_set = np.column_stack((feat_set, res))
        # loc_set = np.column_stack((loc_set, iloc.astype('int')))

print('all feat_set')
feat_set = feat_set.T
print(feat_set.shape)

model = vMFMM(cluster_num, 'k++', tmp_dir='/export/home/qliu24/tmp/vMFMM/set{}/'.format(subset_idx))
model.fit(feat_set, 30, max_it=150)

with open(Dict['Dictionary_sub'].format(cluster_num,subset_idx), 'wb') as fh:
    pickle.dump([model.p, model.mu, model.pi], fh)

############## save examples ###################
# with open(Dict['file_list'], 'r') as fh:
#     image_path = [ff.strip() for ff in fh.readlines()]

# num = 50
# print('save top {0} images for each cluster'.format(num))
# example = [None for vc_i in range(cluster_num)]
# for vc_i in range(cluster_num):
#     patch_set = np.zeros(((Arf**2)*3, num)).astype('uint8')
#     sort_idx = np.argsort(-model.p[:,vc_i])[0:num]
#     for idx in range(num):
#         iloc = loc_set[:,sort_idx[idx]]
#         img = cv2.imread(os.path.join(Dict['file_dir'], image_path[iloc[0]]))
#         img = myresize(img, scale_size, 'short')
        
#         patch = img[iloc[1]:iloc[3], iloc[2]:iloc[4], :]
#         patch_set[:,idx] = patch.flatten()
        
#     example[vc_i] = np.copy(patch_set)
#     if vc_i%10 == 0:
#         print(vc_i)
        
# save_path2 = Dict['Dictionary'].replace('.pickle','_example.pickle')
# with open(save_path2, 'wb') as fh:
#     pickle.dump(example, fh)
