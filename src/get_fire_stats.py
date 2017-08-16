from scipy.spatial.distance import cdist
from config_voting_ILSVRC12 import *

def get_fire_stats(idx, magic_thh, centers):
    filename = os.path.join(Feat['cache_dir'], '{}_feat_{}.pickle'.format(VC['layer'], idx))
    
    with open(filename, 'rb') as fh:
        layer_feature = pickle.load(fh)
        
    
    N = len(layer_feature)
    print('{0}: total number of instances {1}'.format(idx, N))
    print(layer_feature[0].shape)
    
    layer_feature_dist = []
    for nn in range(N):
        iheight,iwidth = layer_feature[nn].shape[0:2]
        lff = layer_feature[nn].reshape(-1, 512)
        lff_norm = lff/np.sqrt(np.sum(lff**2, 1)).reshape(-1,1)
        layer_feature_dist.append(cdist(lff_norm, centers, 'cosine').reshape(iheight,iwidth,-1))
        
    layer_feature_b = [None for nn in range(N)]
    
    for nn in range(N):
        layer_feature_b[nn] = (layer_feature_dist[nn]<magic_thh).astype(int)
        
    vc_fire_cnt = [None for nn in range(N)]
    vc_fire_empty = [None for nn in range(N)]
    for nn in range(N):
        vc_fire_cnt[nn] = np.mean(np.sum(layer_feature_b[nn], axis=2))
        vc_fire_empty[nn] = np.sum(np.sum(layer_feature_b[nn], axis=2)==0)/np.prod(layer_feature_b[nn].shape[0:2])
    
    
    print('{0}, {1}'.format(np.mean(vc_fire_cnt), np.mean(vc_fire_empty)))
    return(np.mean(vc_fire_cnt), np.mean(vc_fire_empty))


subset_idx = 3
cnt_ls = []
empty_ls = []
with open(Dict['Dictionary_sub'].format(VC['num'], subset_idx), 'rb') as fh:
        _, centers, _ = pickle.load(fh)
        
subset_ls = np.where(subset_lb==subset_idx)[0]
for idx in subset_ls:
    cnt, empty = get_fire_stats(idx, 0.49, centers)
    cnt_ls.append(cnt)
    empty_ls.append(empty)
    
print('overall:')
print('{0}, {1}'.format(np.mean(cnt_ls), np.mean(empty_ls)))

