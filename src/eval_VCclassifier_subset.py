from config_voting_ILSVRC12 import *
from scipy.spatial.distance import cdist
from cls_factorizable_funcs import *

subset_idx = 7
magic_thh=0.49
K = 4

dict_file = Dict['Dictionary_sub'].format(VC['num'], subset_idx)
with open(dict_file, 'rb') as fh:
    _,centers,_ = pickle.load(fh)

subset_ls = np.where(subset_lb==subset_idx)[0]
for cls_idx in subset_ls:
    print('class index: {}'.format(cls_idx))

    # load test feat and convert to 0-1 VC encoding
    test_feat = os.path.join(Feat['cache_dir_val'], 'pool4_feat_{}.pickle'.format(cls_idx))
    with open(test_feat,'rb') as fh:
        layer_feature = pickle.load(fh)

    N = len(layer_feature)
    print('Total number of val samples for obj {}: {}'.format(cls_idx, N))
    
    if N > 50:
        N_slt = np.random.permutation(N)[0:50]
        layer_feature = [layer_feature[nn] for nn in N_slt]
        N = len(layer_feature)
        print('Total number of val samples for obj {}: {}'.format(cls_idx, N))

    r_set = [None for nn in range(N)]
    for nn in range(N):
        iheight,iwidth = layer_feature[nn].shape[0:2]
        lff = layer_feature[nn].reshape(-1, featDim)
        lff_norm = lff/np.sqrt(np.sum(lff**2, 1)).reshape(-1,1)
        r_set[nn] = cdist(lff_norm, centers, 'cosine').reshape(iheight,iwidth,-1)

    layer_feature_b = [None for nn in range(N)]
    for nn in range(N):
        layer_feature_b[nn] = (r_set[nn]<magic_thh).astype(int)
    
    # load mixture models
    cls_candidates = np.where(subset_lb==subset_idx)[0]
    cls_total = len(cls_candidates)
    print('Total number of candidate classes: {}'.format(cls_total))
    all_weights = []
    all_logZs = []

    for cc in cls_candidates:
        model_file = os.path.join(Model_dir, 'obj{}_K{}_notrain.pickle'.format(cc, K))
        with open(model_file,'rb') as fh:
            weights, _ = pickle.load(fh)

        assert(len(weights)==K)
        logZs = []
        for kk in range(K):
            # logZs.append(np.sum(np.log(1+np.exp(weights[kk]))))
            logZs.append(np.log(1+np.exp(weights[kk])))

        all_weights.append(weights)
        all_logZs.append(logZs)

    all_scores = np.zeros((N, cls_total))
    for nn in range(N):
        print(nn,end=' ', flush=True)
        for cc in range(cls_total):
            all_scores[nn,cc] = comptScoresM(layer_feature_b[nn], all_weights[cc], all_logZs[cc])

    print('')
    rst = np.argmax(all_scores, axis=1)
    accu = np.sum(rst==np.where(cls_candidates==cls_idx)[0][0])/N
    print('accuracy is: %2.2f'% accu)
    
    rst_file = os.path.join(Result_dir, 'scores_obj{}.pickle'.format(cls_idx))
    with open(rst_file, 'wb') as fh:
        pickle.dump(all_scores, fh)
