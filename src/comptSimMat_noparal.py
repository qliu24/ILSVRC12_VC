from scipy.spatial.distance import cdist
from vcdist_funcs import *
from config_voting_ILSVRC12 import *
import time

np.random.seed(0)

inst_per_cat = 30
savename = Feat['cache_dir'] + 'simmat_mthrh047.pickle'
magic_thh = 0.47

layer_feature = []
inst_idx_ls = []
for ii in range(127):
    print(ii)
    fname = os.path.join(Feat['cache_dir'], '{}_feat_{}.pickle'.format(VC['layer'], ii))
    with open(fname, 'rb') as fh:
        layer_feature_i = pickle.load(fh)
        
    idx_sl = np.random.permutation(len(layer_feature_i))[0:inst_per_cat]
    layer_feature += [layer_feature_i[idd] for idd in idx_sl]
    inst_idx_ls.append(idx_sl)
    
N = len(layer_feature)
print('total number of instances {0}'.format(N))

with open(Dict['Dictionary'], 'rb') as fh:
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

mat_dis1 = np.ones((N,N))
mat_dis2 = np.ones((N,N))
for nn in range(N):
    print(nn, end=' ', flush=True)
        
    rst1, rst2 = vc_dis_paral((layer_feature_b, nn))
    mat_dis1[nn] = rst1
    mat_dis2[nn] = rst2

_e = time.time()
print('comptSimMat total time: {}'.format((_e-_s)/60))

with open(savename, 'wb') as fh:
    pickle.dump([mat_dis1, mat_dis2], fh)