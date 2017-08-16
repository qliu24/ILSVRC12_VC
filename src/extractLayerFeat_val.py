from scipy.spatial.distance import cdist
from FeatureExtractor import *
from config_voting_ILSVRC12 import *

val_file_ls = '/export/home/qliu24/dataset/ILSVRC12/list/val_wn.txt'
with open(val_file_ls,'r') as fh:
    contents = fh.readlines()
    
filenames = np.array([cc.strip().split()[0] for cc in contents])
cls_labels = np.array([cc.strip().split()[1] for cc in contents], dtype='int')

extractor = FeatureExtractor(cache_folder=model_cache_folder, which_net='vgg16', which_layer=VC['layer'], which_snapshot=0)

for ii in range(127):
    files_ii = filenames[cls_labels==ii]
    Nii = len(files_ii)
    print('Number of val images for obj {}: {}'.format(ii, Nii))
    feat_set = []
    for nn in range(Nii):
        if not os.path.isfile(os.path.join(Dataset['img_dir_val'], files_ii[nn])):
            continue
            
        file_img = os.path.join(Dataset['img_dir_val'], files_ii[nn])
        assert(os.path.isfile(file_img))
        img = cv2.imread(file_img)
        patch = myresize(img, scale_size, 'short')
        
        layer_feature = extractor.extract_feature_image(patch)[0]
        assert(featDim == layer_feature.shape[2])
        feat_set.append(layer_feature)
        
        if nn%100 == 0:
            print(nn, end=' ', flush=True)
            
    print('')
    print('Number of val fg images for obj {}: {}'.format(ii, len(feat_set)))
    
    file_cache_feat = os.path.join(Feat['cache_dir'], 'val','{0}_feat_{1}.pickle'.format(VC['layer'], ii))
    with open(file_cache_feat, 'wb') as fh:
        pickle.dump(feat_set, fh)