from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
from vcdist_funcs import *
from config_voting_ILSVRC12 import *
import time

np.random.seed(0)
subset_idx = 0
subset_ls = np.where(subset_lb==subset_idx)[0]

paral_num = 30
inst_per_cat = 45
savename = os.path.join(Feat['cache_dir'],'simmat_mthrh048_set{}.pickle'.format(subset_idx))
magic_thh = 0.48

layer_feature = []
for ii in subset_ls:
    print(ii, end=' ',flush=True)
    fname = os.path.join(Feat['cache_dir'], '{}_feat_{}.pickle'.format(VC['layer'], ii))
    with open(fname, 'rb') as fh:
        layer_feature_i = pickle.load(fh)
        
    # idx_sl = np.random.permutation(len(layer_feature_i))[0:inst_per_cat]
    # layer_feature += [layer_feature_i[idd] for idd in idx_sl]
    layer_feature_i = layer_feature_i[0:inst_per_cat]
    layer_feature += layer_feature_i
    
print('')
'''
fname = os.path.join(Feat['cache_dir'], '{}_feat_all_30PerObj.pickle'.format(VC['layer']))
with open(fname, 'rb') as fh:
    layer_feature = pickle.load(fh)
    
'''

N = len(layer_feature)
print('total number of instances {0}'.format(N))

# N = 210
# layer_feature = layer_feature[0:N]

with open(Dict['Dictionary_sub'].format(VC['num_sub'][subset_idx], subset_idx), 'rb') as fh:
    _, centers, _ = pickle.load(fh)

r_set = [None for nn in range(N)]
for nn in range(N):
    iheight,iwidth = layer_feature[nn].shape[0:2]
    lff = layer_feature[nn].reshape(-1, 512)
    lff_norm = lff/np.sqrt(np.sum(lff**2, 1)).reshape(-1,1)
    r_set[nn] = cdist(lff_norm, centers, 'cosine').reshape(iheight,iwidth,-1)

layer_feature_b = [None for nn in range(N)]
for nn in range(N):
    layer_feature_b[nn] = (r_set[nn]<magic_thh).astype(int).T

print('Start compute sim matrix ...')
_s = time.time()
'''
inputs = [(layer_feature_b, nn) for nn in range(N)]
para_rst = np.array(Parallel(n_jobs=paral_num)(delayed(vc_dis_paral)(i) for i in inputs))
mat_dis1 = para_rst[:,0]
mat_dis2 = para_rst[:,1]
'''
mat_dis1 = np.ones((N,N))
mat_dis2 = np.ones((N,N))
N_sub = 200
sub_cnt = int(math.ceil(N/N_sub))
for ss1 in range(sub_cnt):
    start1 = ss1*N_sub
    end1 = min((ss1+1)*N_sub, N)
    layer_feature_b_ss1 = layer_feature_b[start1:end1]
    for ss2 in range(ss1,sub_cnt):
        print('iter {1}/{0} {2}/{0}'.format(sub_cnt, ss1+1, ss2+1))
        _ss = time.time()
        start2 = ss2*N_sub
        end2 = min((ss2+1)*N_sub, N)
        if ss1==ss2:
            inputs = [(layer_feature_b_ss1, nn) for nn in range(end2-start2)]
            para_rst = np.array(Parallel(n_jobs=paral_num)(delayed(vc_dis_paral)(i) for i in inputs))
            
        else:
            layer_feature_b_ss2 = layer_feature_b[start2:end2]
            inputs = [(layer_feature_b_ss2, lfb) for lfb in layer_feature_b_ss1]
            para_rst = np.array(Parallel(n_jobs=paral_num)(delayed(vc_dis_paral_full)(i) for i in inputs))
            
        mat_dis1[start1:end1, start2:end2] = para_rst[:,0]
        mat_dis2[start1:end1, start2:end2] = para_rst[:,1]
        _ee = time.time()
        print('comptSimMat iter time: {}'.format((_ee-_ss)/60))


_e = time.time()
print('comptSimMat total time: {}'.format((_e-_s)/60))

with open(savename, 'wb') as fh:
    pickle.dump([mat_dis1, mat_dis2], fh)