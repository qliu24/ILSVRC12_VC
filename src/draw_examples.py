import pickle
import numpy as np
import cv2

fname='/export/home/qliu24/ILSVRC12_VC/dictionary/dictionary_ILSVRC12_VGG16_pool4_K512_vMFMM30_example.pickle'
with open(fname,'rb') as fh:
    example = pickle.load(fh)

print(len(example))

Arf = 100

for ii in range(len(example)):
    big_img = np.zeros((10+(Arf+10)*4, 10+(Arf+10)*5, 3))
    for iis in range(20):
        if iis >= example[ii].shape[1]:
            continue

        aa = iis//5
        bb = iis%5
        rnum = 10+aa*(Arf+10)
        cnum = 10+bb*(Arf+10)
        big_img[rnum:rnum+Arf, cnum:cnum+Arf, :] = example[ii][:,iis].reshape(Arf,Arf,3).astype('uint8')

    fname = '/export/home/qliu24/ILSVRC12_VC/dictionary/ILSVRC12_K512/example_K' + str(ii) + '.png'
    cv2.imwrite(fname, big_img)

