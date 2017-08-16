from scipy.spatial.distance import cdist
from FeatureExtractor import *
from config_voting_ILSVRC12 import *

def extractLayerFeat(idx_ls, scale_size=224):
    extractor = FeatureExtractor(cache_folder=model_cache_folder, which_net='vgg16', which_layer=VC['layer'], which_snapshot=0)
    '''
    assert(os.path.isfile(Dict['Dictionary']))
    with open(Dict['Dictionary'], 'rb') as fh:
        _,centers,_ = pickle.load(fh)
    '''
    for ii in idx_ls:
        file_list = Dataset['file_list'].format(ii)
        with open(file_list, 'r') as fh:
            content = fh.readlines()
            
        img_list = [x.strip() for x in content]
        img_num = len(img_list)
        print('total number of images for idx {1}: {0}'.format(img_num, ii))
        
        img_list = img_list[0:3000]
        img_num = len(img_list)
        print('used number of images for idx {1}: {0}'.format(img_num, ii))
        
        feat_set = [None for nn in range(img_num)]
        for nn in range(img_num):
            file_img = os.path.join(Dataset['img_dir'], img_list[nn])
            assert(os.path.isfile(file_img))
            img = cv2.imread(file_img)
            patch = myresize(img, scale_size, 'short')
            
            layer_feature = extractor.extract_feature_image(patch)[0]
            assert(featDim == layer_feature.shape[2])
            feat_set[nn] = layer_feature
            '''
            iheight, iwidth = layer_feature.shape[0:2]
            layer_feature = layer_feature.reshape(-1, featDim)
            feat_norm = np.sqrt(np.sum(layer_feature**2, 1)).reshape(-1,1)
            layer_feature = layer_feature/feat_norm
            
            dist = cdist(layer_feature, centers, 'cosine').reshape(iheight,iwidth,-1)
            assert(dist.shape[2]==centers.shape[0]);
            r_set[nn] = dist
            '''
            if nn%100 == 0:
                print(nn, end=' ')
                sys.stdout.flush()
            
            
        print('\n')
        
        file_cache_feat = os.path.join(Feat['cache_dir'], '{0}_feat_{1}.pickle'.format(VC['layer'], ii))
        with open(file_cache_feat, 'wb') as fh:
            pickle.dump(feat_set, fh)
            
            
if __name__=='__main__':
    # objs = ['car','aeroplane','bicycle','bus','motorbike','train']
    idx_ls = list(range(127))
    extractLayerFeat(idx_ls)