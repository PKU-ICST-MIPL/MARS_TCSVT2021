import scipy.io as sio
import h5py
import numpy as np

def load_deep_features(data_name):
    valid_data = True

    if data_name == 'xmedia':
        path = 'datasets/XMedia/XMediaFeatures.mat'
        all_data = sio.loadmat(path)
        I_te_CNN = all_data['I_te_CNN'].astype('float32')   # Features of test set for image data, CNN feature
        I_tr_CNN = all_data['I_tr_CNN'].astype('float32')   # Features of training set for image data, CNN feature
        T_te_BOW = all_data['T_te_BOW'].astype('float32')   # Features of test set for text data, BOW feature
        T_tr_BOW = all_data['T_tr_BOW'].astype('float32')   # Features of training set for text data, BOW feature
        V_te_CNN = all_data['V_te_CNN'].astype('float32')   # Features of test set for video(frame) data, CNN feature
        V_tr_CNN = all_data['V_tr_CNN'].astype('float32')   # Features of training set for video(frame) data, CNN feature
        A_te = all_data['A_te'].astype('float32')           # Features of test set for audio data, MFCC feature
        A_tr = all_data['A_tr'].astype('float32')           # Features of training set for audio data, MFCC feature
        d3_te = all_data['d3_te'].astype('float32')         # Features of test set for 3D data, LightField feature
        d3_tr = all_data['d3_tr'].astype('float32')         # Features of training set for 3D data, LightField feature

        teImgCat = all_data['teImgCat'].reshape([-1]).astype('int64') # category label of test set for image data
        trImgCat = all_data['trImgCat'].reshape([-1]).astype('int64') # category label of training set for image data
        teVidCat = all_data['teVidCat'].reshape([-1]).astype('int64') # category label of test set for video(frame) data
        trVidCat = all_data['trVidCat'].reshape([-1]).astype('int64') # category label of training set for video(frame) data
        teTxtCat = all_data['teTxtCat'].reshape([-1]).astype('int64') # category label of test set for text data
        trTxtCat = all_data['trTxtCat'].reshape([-1]).astype('int64') # category label of training set for text data
        te3dCat = all_data['te3dCat'].reshape([-1]).astype('int64')   # category label of test set for 3D data
        tr3dCat = all_data['tr3dCat'].reshape([-1]).astype('int64')   # category label of training set for 3D data
        teAudCat = all_data['teAudCat'].reshape([-1]).astype('int64') # category label of test set for audio data
        trAudCat = all_data['trAudCat'].reshape([-1]).astype('int64') # category label of training set for audio data

        train_data = [I_tr_CNN, T_tr_BOW, A_tr, d3_tr, V_tr_CNN]
        test_data = [I_te_CNN[0: 500], T_te_BOW[0: 500], A_te[0: 100], d3_te[0: 50], V_te_CNN[0: 87]]
        valid_data = [I_te_CNN[500::], T_te_BOW[500::], A_te[100::], d3_te[50::], V_te_CNN[87::]]
        train_labels = [trImgCat, trTxtCat, trAudCat, tr3dCat, trVidCat]
        test_labels = [teImgCat[0: 500], teTxtCat[0: 500], teAudCat[0: 100], te3dCat[0: 50], teVidCat[0: 87]]
        valid_labels = [teImgCat[500::], teTxtCat[500::], teAudCat[100::], te3dCat[50::], teVidCat[87::]]

    elif data_name == 'xmedianet':
        path = 'datasets/XMediaNet/XMediaNetFeatures.mat'
        all_data = sio.loadmat(path)
        I_te_CNN = all_data['I_te'].astype('float32')   # Features of test set for image data, CNN feature
        I_tr_CNN = all_data['I_tr'].astype('float32')   # Features of training set for image data, CNN feature
        T_te_BOW = all_data['T_te'].astype('float32')   # Features of test set for text data, BOW feature
        T_tr_BOW = all_data['T_tr'].astype('float32')   # Features of training set for text data, BOW feature
        V_te_CNN = all_data['V_te'].astype('float32')   # Features of test set for video(frame) data, CNN feature
        V_tr_CNN = all_data['V_tr'].astype('float32')   # Features of training set for video(frame) data, CNN feature
        A_te = all_data['A_te'].astype('float32')           # Features of test set for audio data, MFCC feature
        A_tr = all_data['A_tr'].astype('float32')           # Features of training set for audio data, MFCC feature
        d3_te = all_data['TD_te'].astype('float32')         # Features of test set for 3D data, LightField feature
        d3_tr = all_data['TD_tr'].astype('float32')         # Features of training set for 3D data, LightField feature        

        I_labs = all_data['I_labs'].reshape([-1]).astype('int64')
        T_labs = all_data['T_labs'].reshape([-1]).astype('int64')
        V_labs = all_data['V_labs'].reshape([-1]).astype('int64')
        A_labs = all_data['A_labs'].reshape([-1]).astype('int64')
        d3_labs = all_data['TD_labs'].reshape([-1]).astype('int64')

        train_data = [I_tr_CNN[0:5000], T_tr_BOW[0:5000], V_tr_CNN[0:5000], A_tr[0:5000], d3_tr]
        valid_data = [I_te_CNN[0: 500], T_te_BOW[0: 500], V_te_CNN[0: 500], A_te[0: 500], d3_te[0: 200]]
        test_data = [I_te_CNN[500:1000], T_te_BOW[500:1000], V_te_CNN[500:1000], A_te[500:1000], d3_te[200::]]
        train_labels = [I_labs[0:5000], T_labs[0:5000], V_labs[0:5000], A_labs[0:5000], d3_labs[0:1600]]
        valid_labels = [I_labs[32000: 32500], T_labs[32000: 32500], V_labs[8000: 8500], A_labs[8000: 8500], d3_labs[1600: 1800]]
        test_labels = [I_labs[32500:33000], T_labs[32500:33000], V_labs[8500:9000], A_labs[8500:9000], d3_labs[1800:2000]]

    elif data_name == 'wiki':
        valid_len = 231
        path = 'datasets/Wiki/wiki.mat'
        all_data = sio.loadmat(path)
        train_imgs = all_data['train_imgs_deep'].astype('float32')
        train_imgs_labels = all_data['train_imgs_labels'].reshape([-1]).astype('int64')
        train_texts = all_data['train_texts_doc'].astype('float32')
        train_texts_labels = all_data['train_texts_labels'].reshape([-1]).astype('int64')

        test_imgs = all_data['test_imgs_deep'].astype('float32')
        test_imgs_labels = all_data['test_imgs_labels'].reshape([-1]).astype('int64')
        test_texts = all_data['test_texts_doc'].astype('float32')
        test_texts_labels = all_data['test_texts_labels'].reshape([-1]).astype('int64')

        train_data = [train_imgs, train_texts]
        train_labels = [train_imgs_labels, train_texts_labels]
        test_data = [test_imgs, test_texts]
        test_labels = [test_imgs_labels, test_texts_labels]

        valid_data = [test_data[0][0: valid_len], test_data[1][0: valid_len]]
        valid_labels = [test_labels[0][0: valid_len], test_labels[1][0: valid_len]]
        test_data = [test_data[0][valid_len:], test_data[1][valid_len:]]
        test_labels = [test_labels[0][valid_len:], test_labels[1][valid_len:]]
 
    elif data_name == 'nus21':
        path = 'datasets/NUS-WIDE/nus-wide-tc21.mat'
        all_data = sio.loadmat(path)
        img_train = all_data['img_train'].astype('float32')
        img_test = all_data['img_test'].astype('float32')
        img_val = all_data['img_val'].astype('float32')

        txt_train = all_data['txt_train'].astype('float32')
        txt_test = all_data['txt_test'].astype('float32')
        txt_val = all_data['txt_val'].astype('float32')

        train_labs = all_data['train_labs'].astype('int64')
        test_labs = all_data['test_labs'].astype('int64')
        val_labs = all_data['val_labs'].astype('int64')

        train_data = [img_train, txt_train]
        test_data = [img_test, txt_test]
        valid_data = [img_val, txt_val]
        train_labels = [train_labs, train_labs]
        test_labels = [test_labs, test_labs]
        valid_labels =  [val_labs, val_labs]

    if valid_data:
        return train_data, train_labels, valid_data, valid_labels, test_data, test_labels
    else:
        return train_data, train_labels, test_data, test_labels
