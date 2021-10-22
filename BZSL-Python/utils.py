import numpy as np
import scipy.io as sio
from scipy.linalg import eigh
import os

"""Data loading part"""
class data_loader(object):
    def __init__(self, datapath, dataset, side_info='original', tuning=False):

        print("The current working directory is")
        print(os.getcwd())
        self.datapath =  datapath  # '../data/'
        self.dataset = dataset
        self.side_info_source = side_info
        self.tuning = tuning

        self.read_matdata()


    def read_matdata(self):
        path = os.path.join(self.datapath, self.dataset, 'res101.mat')
        data_mat = sio.loadmat(path)
        self.features = data_mat['features'].T
        self.labels = data_mat['labels'].ravel() - 1

        path = os.path.join(self.datapath, self.dataset, 'att_splits.mat')
        splits_mat = sio.loadmat(path)

        self.trainval_loc = splits_mat['trainval_loc'].ravel() - 1
        self.train_loc = splits_mat['train_loc'].ravel() - 1 
        self.val_unseen_loc = splits_mat['val_loc'].ravel() - 1 
        self.test_seen_loc = splits_mat['test_seen_loc'].ravel() - 1
        self.test_unseen_loc = splits_mat['test_unseen_loc'].ravel() - 1


        if self.side_info_source not in ['original', 'w2v', 'dna']:
            print('Please choose a valid source for side information. There are 3 possibilities for CUB data: ["original", "w2v", "dna"] and one for INSECT: "dna"')
            return 

        if (self.dataset=='INSECT') and (self.side_info_source!='dna'):
            print('Invalid side information source for INSECT data! There is only one side information source for INSECT dataset: "dna". Model will continue using DNA as side information')

        self.side_info = splits_mat['att']
        if self.side_info_source=='w2v':
            self.side_info = splits_mat['att_w2v']
        elif self.side_info_source=='dna':
            self.side_info = splits_mat['att_dna']
        

    def data_split(self):

        train_idx = self.trainval_loc
        test_seen_idx = self.test_seen_loc
        test_unseen_idx = self.test_unseen_loc

        if self.tuning:
            train_idx = self.train_loc
            test_seen_idx = self.val_seen_loc
            test_unseen_idx = self.val_unseen_loc
        
        xtrain = self.features[train_idx]
        ytrain = self.labels[train_idx]
        xtest_seen = self.features[test_seen_idx]
        ytest_seen = self.labels[test_seen_idx]
        xtest_unseen = self.features[test_unseen_idx]
        ytest_unseen = self.labels[test_unseen_idx]

        self.seenclasses = np.unique(ytrain)
        self.unseenclasses = np.unique(ytest_unseen)

        return xtrain, ytrain, xtest_seen, ytest_seen, xtest_unseen, ytest_unseen

    def load_tuned_params(self):

        if self.dataset not in ['INSECT', 'CUB']:
            print('The provided dataset is not in the gallery. Please use one of these 2 datsets to load tuned params: ["INSECT", "CUB"]')
            return

        dim  = 500
        
        if self.dataset=='INSECT':
            hyperparams  = [0.1, 10, 5*dim, 10, 3]

        if self.dataset=='CUB':
            if self.side_info_source=='original':
                hyperparams  = [1, 25, 500*dim, 10, 3]
            elif self.side_info_source=='w2v':
                hyperparams  = [0.1, 25, 5*dim, 5, 2]
            elif self.side_info_source=='dna':
                hyperparams  = [0.1, 25, 25*dim, 5, 3]

        return self.side_info, hyperparams[0], hyperparams[1], hyperparams[2], hyperparams[3], hyperparams[4]


### Seen, Unseen class and Harmonic mean claculation ###
def perf_calc_acc(y_ts_s, y_ts_us, ypred_s, ypred_us):
    seen_cls = np.unique(y_ts_s)
    unseen_cls = np.unique(y_ts_us)
    # Performance calculation
    acc_per_cls_s = np.zeros((len(seen_cls), 1))
    acc_per_cls_us = np.zeros((len(unseen_cls), 1))

    for i in range(len(seen_cls)):
        lb = seen_cls[i]
        idx = y_ts_s==lb
        acc_per_cls_s[i] = np.sum(ypred_s[idx.ravel()]==lb)/np.sum(idx)

    for i in range(len(unseen_cls)):
        lb = unseen_cls[i]
        idx = y_ts_us==lb
        acc_per_cls_us[i] = np.sum(ypred_us[idx.ravel()]==lb)/np.sum(idx)

    ave_s = np.mean(acc_per_cls_s)
    ave_us = np.mean(acc_per_cls_us)
    H = 2*ave_s*ave_us/(ave_s+ave_us)
    
    return acc_per_cls_s, acc_per_cls_us, ave_s, ave_us, H

def apply_pca(x_tr, x_ts_s, x_ts_us, pca_dim):
    # Dimentionality reduction using PCA
    _, eig_vec = eigh(np.cov(x_tr.T))
    x_tr    = np.dot(x_tr, eig_vec[:, -pca_dim:])
    x_ts_s    = np.dot(x_ts_s, eig_vec[:, -pca_dim:])
    x_ts_us    = np.dot(x_ts_us, eig_vec[:, -pca_dim:])

    return x_tr, x_ts_s, x_ts_us