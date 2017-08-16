import cv2,os,glob,pickle,sys,math,time
import numpy as np
from myresize import myresize

net_type='VGG'
# Alexnet
if net_type=='alex':
    Apad_set = [0, 0, 16, 16, 32, 48, 64] # padding size
    Astride_set = [4, 8, 8, 16, 16, 16, 16] # stride size
    featDim_set = [96, 96, 256, 256, 384, 384, 256] # feature dimension
    Arf_set = [11, 19, 51, 67, 99, 131, 163]
    offset_set = np.ceil(np.array(Apad_set)/np.array(Astride_set)).astype(int)
    layer_n = 4 # conv3
elif net_type=='VGG':
    Apad_set = [2, 6, 18, 42, 90] # padding size
    Astride_set = [2, 4, 8, 16, 32] # stride size
    featDim_set = [64, 128, 256, 512, 512] # feature dimension
    Arf_set = [6, 16, 44, 100, 212]
    offset_set = np.ceil(np.array(Apad_set).astype(float)/np.array(Astride_set)).astype(int)
    layer_n = 3 # pool4
    # layer_n = 2 # pool3
    # layer_n = 1 # pool2
    
Apad = Apad_set[layer_n]
Astride = Astride_set[layer_n]
featDim = featDim_set[layer_n]
Arf = Arf_set[layer_n]
offset = offset_set[layer_n]

scale_size = 224

VC = dict()
VC['num'] = 200

if net_type=='alex':
    VC['layer'] = 'conv3'
elif net_type=='VGG':
    VC['layer'] = 'pool4'
    # VC['layer'] = 'pool3'
    # VC['layer'] = 'pool2'

model_cache_folder = '/export/home/qliu24/qing_voting_139/qing_voting_py/cache/'
root_dir = '/export/home/qliu24/ILSVRC12_VC/'

Dict = dict()
Dict['file_list'] = '/export/home/qliu24/dataset/ILSVRC12/list_fg/file_list_200.txt'
Dict['cache_path'] = '/export/home/qliu24/ILSVRC12_VC/feat/round1/{0}_all_dumped_data'.format(VC['layer'])
Dict['cache_path_sub'] = '/export/home/qliu24/ILSVRC12_VC/feat/round3/{0}_all_dumped_data'.format(VC['layer'])
Dict['file_dir'] = '/export/home/qliu24/dataset/ILSVRC12/ILSVRC2012/train_fg/'
Dict['Dictionary'] = '/export/home/qliu24/ILSVRC12_VC/dictionary/dictionary_ILSVRC12_VGG16_pool4_K512_vMFMM30.pickle'
Dict['Dictionary_sub'] = '/export/home/qliu24/ILSVRC12_VC/dictionary/dictionary_ILSVRC12_VGG16_pool4_K{}_vMFMM30_set{}.pickle'

Dataset = dict()
Dataset['file_list'] = '/export/home/qliu24/dataset/ILSVRC12/list_fg/train_wn_rndm_{}.txt'
Dataset['file_list_val'] = '/export/home/qliu24/dataset/ILSVRC12/list/val_wn.txt'
Dataset['img_dir'] = '/export/home/qliu24/dataset/ILSVRC12/ILSVRC2012/train_fg/'
Dataset['img_dir_val'] = '/export/home/qliu24/dataset/ILSVRC12/ILSVRC2012/val_fg/'

Feat = dict()
Feat['cache_dir'] = os.path.join(root_dir, 'feat')
Feat['cache_dir_val'] = os.path.join(root_dir, 'feat', 'val')
if not os.path.exists(Feat['cache_dir']):
    os.makedirs(Feat['cache_dir'])
    
subset_lb2=np.array([0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 3, 0, 0, 2, 0, 0, 0, 0, 3, 2, 1, 0, 3, 0, 0, 0, 0, 0, 2, 0, 0, 2, 2, 0, 3, 0, 3, 2, 3, 3, 0, 3, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 0, 1, 3, 2, 0, 0, 3, 1, 3, 0, 0, 0, 0, 0, 2, 0, 2, 0, 1, 3, 2, 2, 0, 2, 1, 0, 0, 0, 0, 0, 0, 3, 0, 0, 2, 1, 0, 3, 2, 0, 2, 0, 0, 0, 0, 3, 2, 2, 0, 3, 1, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0],dtype=int)

subset_lb=np.array([0, 7, 5, 7, 2, 2, 7, 5, 0, 7, 4, 2, 0, 5, 0, 2, 3, 0, 6, 2, 7, 0, 6, 4, 3, 2, 1, 6, 3, 7, 5, 0, 6, 6, 2, 7, 6, 2, 2, 5, 3, 6, 3, 2, 3, 3, 5, 3, 5, 2, 6, 5, 6, 0, 2, 6, 7, 4, 0, 4, 7, 6, 3, 3, 3, 0, 1, 3, 2, 6, 6, 3, 1, 3, 4, 7, 7, 5, 5, 2, 6, 2, 6, 1, 3, 2, 2, 6, 2, 1, 5, 4, 6, 0, 6, 6, 3, 0, 7, 2, 1, 0, 3, 2, 6, 2, 0, 5, 6, 4, 3, 2, 2, 0, 3, 1, 0, 5, 5, 3, 5, 4, 7, 5, 7, 6, 6],dtype=int)

subset_cnt = [16,7,23,20,8,16,23,14]

with open('/export/home/qliu24/dataset/ILSVRC12/list_fg/cls_labels_127.txt','r') as fh:
    content_txt = fh.readlines()
    
cls_lb_127 = [cc.strip().split()[1] for cc in content_txt]

Model_dir = os.path.join(root_dir, 'mix_model')
Result_dir = os.path.join(root_dir, 'result')