from config_voting_ILSVRC12 import *

np.random.seed(0)
subset_idx = 1
subset_ls = np.where(subset_lb==subset_idx)[0]

inst_per_cat = 400
savename = os.path.join(Feat['cache_dir'],'pool4_feat_set{}_{}PerObj.pickle'.format(subset_idx,inst_per_cat))

layer_feature = []
inst_idx_ls = []
for ii in subset_ls:
    print(ii, end=' ',flush=True)
    fname = os.path.join(Feat['cache_dir'], '{}_feat_{}.pickle'.format(VC['layer'], ii))
    with open(fname, 'rb') as fh:
        layer_feature_i = pickle.load(fh)
        
    idx_sl = np.random.permutation(len(layer_feature_i))[0:inst_per_cat]
    layer_feature += [layer_feature_i[idd] for idd in idx_sl]
    inst_idx_ls.append(idx_sl)
    
print('')

with open(savename, 'wb') as fh:
    pickle.dump([layer_feature, inst_idx_ls], fh)

