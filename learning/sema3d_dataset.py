"""
    Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs
    http://arxiv.org/abs/1711.09869
    2017 Loic Landrieu, Martin Simonovsky
"""
from __future__ import division
from __future__ import print_function
from builtins import range

import random
import numpy as np
import os
import functools
import torch
import torchnet as tnt
import h5py
import spg


def get_datasets(args, test_seed_offset=0):
    train_names = ['6755_66525.h5', '6955_66645.h5', '6875_66585.h5', '6820_66550.h5', '6950_66640.h5', '6955_66650.h5', '7055_66455.h5', '7020_66300.h5', '7050_66335.h5', '6825_66560.h5', '7015_66550.h5', '7055_66350.h5', '7030_66525.h5', '6985_66650.h5', '7005_66575.h5', '6750_66520.h5', '6770_66530.h5', '6990_66595.h5', '6760_66530.h5', '6805_66540.h5', '7000_66580.h5', '6980_66650.h5', '7060_66385.h5', '7055_66410.h5', '7010_66565.h5', '7015_66555.h5', '6810_66545.h5', '7040_66495.h5', '6870_66580.h5', '7005_66565.h5', '7045_66485.h5', '7055_66445.h5', '7025_66530.h5', '6835_66565.h5', '7050_66340.h5', '7030_66315.h5', '7035_66325.h5', '6985_66605.h5', '7020_66305.h5', '7040_66505.h5', '6875_66580.h5', '6965_66630.h5', '7015_66560.h5', '7045_66500.h5', '7055_66340.h5', '7060_66420.h5', '6995_66575.h5', '7060_66400.h5', '6800_66540.h5', '7015_66545.h5', '7040_66325.h5', '7020_66310.h5', '7050_66330.h5', '7055_66420.h5', '6900_66605.h5', '6890_66600.h5', '6830_66560.h5', '6790_66540.h5', '6840_66565.h5', '6915_66610.h5', '7020_66540.h5', '7000_66570.h5', '6855_66570.h5', '7045_66335.h5', '6785_66535.h5', '6870_66585.h5', '6950_66645.h5', '6845_66570.h5', '6980_66610.h5', '6990_66650.h5', '6905_66610.h5', '6860_66575.h5', '7060_66360.h5', '6940_66635.h5', '6765_66530.h5', '6935_66635.h5', '6780_66530.h5', '6975_66650.h5', '7055_66345.h5', '7035_66520.h5', '6970_66620.h5', '7005_66570.h5', '6905_66605.h5', '6935_66625.h5', '7020_66550.h5', '7060_66395.h5', '6930_66625.h5', '7050_66450.h5', '7025_66305.h5', '7050_66475.h5', '6900_66600.h5', '6850_66565.h5', '6865_66575.h5', '7045_66495.h5', '6990_66590.h5', '6795_66540.h5', '7035_66320.h5', '7055_66460.h5', '6975_66615.h5', '6910_66610.h5', '7060_66380.h5', '7055_66335.h5', '6845_66565.h5', '6965_66635.h5', '6775_66530.h5', '6750_66525.h5', '6815_66545.h5', '6975_66620.h5']#'6875_66590.h5',, '6960_66635.h5'
    valid_names = ['6835_66560.h5', '6960_66640.h5', '6880_66595.h5', '6860_66580.h5', '6955_66640.h5', '6795_66535.h5']
    if args.db_train_name == 'train':
        trainset = ['train/' + f for f in train_names]
    elif args.db_train_name == 'trainval':
        trainset = ['train/' + f for f in train_names + valid_names]

    validset = []
    testset = []
    if args.use_val_set:
        validset = ['train/' + f for f in valid_names]
    if args.db_test_name == 'testred':
        testset = ['test_reduced/' + os.path.splitext(f)[0] for f in os.listdir(args.SEMA3D_PATH + '/superpoint_graphs/test_reduced')]
    elif args.db_test_name == 'testfull':
        testset = ['test_full/' + os.path.splitext(f)[0] for f in os.listdir(args.SEMA3D_PATH + '/superpoint_graphs/test_full')]
        
    # Load superpoints graphs
    testlist, trainlist, validlist = [], [],  []
    for n in trainset:
        trainlist.append(spg.spg_reader(args, args.SEMA3D_PATH + '/superpoint_graphs/' + n, True))
    for n in validset:
        validlist.append(spg.spg_reader(args, args.SEMA3D_PATH + '/superpoint_graphs/' + n, True))
    for n in testset:
        testlist.append(spg.spg_reader(args, args.SEMA3D_PATH + '/superpoint_graphs/' + n + '.h5', True))

    # Normalize edge features
    if args.spg_attribs01:
        trainlist, testlist, validlist, scaler = spg.scaler01(trainlist, testlist, validlist=validlist)

    return tnt.dataset.ListDataset([spg.spg_to_igraph(*tlist) for tlist in trainlist],
                                    functools.partial(spg.loader, train=True, args=args, db_path=args.SEMA3D_PATH)), \
           tnt.dataset.ListDataset([spg.spg_to_igraph(*tlist) for tlist in testlist],
                                    functools.partial(spg.loader, train=False, args=args, db_path=args.SEMA3D_PATH, test_seed_offset=test_seed_offset)), \
           tnt.dataset.ListDataset([spg.spg_to_igraph(*tlist) for tlist in validlist],
                                    functools.partial(spg.loader, train=False, args=args, db_path=args.SEMA3D_PATH, test_seed_offset=test_seed_offset)),\
            scaler

    
def get_info(args):
    edge_feats = 0
    for attrib in args.edge_attribs.split(','):
        a = attrib.split('/')[0]
        if a in ['delta_avg', 'delta_std', 'xyz']:
            edge_feats += 3
        else:
            edge_feats += 1
    if args.loss_weights == 'none':
        weights = np.ones((6,),dtype='f4')
    else:
        weights = h5py.File(args.SEMA3D_PATH + "/parsed/class_count.h5")["class_count"][:].astype('f4')
        weights = weights.mean()/weights
    if args.loss_weights == 'sqrt':
        weights = np.sqrt(weights)
    weights = torch.from_numpy(weights).cuda() if args.cuda else torch.from_numpy(weights)
    return {
        'node_feats': 14 if args.pc_attribs=='' else len(args.pc_attribs),
        'edge_feats': edge_feats,
        'class_weights': weights,
        'classes': 6,
        'inv_class_map': {0:'terrain', 1:'vegetation', 2:'noise_lower', 3:'ledninger', 4:'crossbeam', 5:'noise_upper'},
    }

def preprocess_pointclouds(SEMA3D_PATH):
    """ Preprocesses data by splitting them by components and normalizing."""
    class_count = np.zeros((6,),dtype='int')
    for n in ['train', 'test_reduced', 'test_full']:
        pathP = '{}/parsed/{}/'.format(SEMA3D_PATH, n)
        if args.supervised_partition :
            pathD = '{}/features_supervision/{}/'.format(SEMA3D_PATH, n)
        else:
            pathD = '{}/features/{}/'.format(SEMA3D_PATH, n)
        pathC = '{}/superpoint_graphs/{}/'.format(SEMA3D_PATH, n)
        if not os.path.exists(pathP):
            os.makedirs(pathP)
        random.seed(0)

        for file in os.listdir(pathC):
            print(file)
            if file.endswith(".h5"):
                f = h5py.File(pathD + file, 'r')

                if n == 'train':
                    labels = f['labels'][:]
                    hard_labels = np.argmax(labels[:,1:],1)
                    label_count = np.bincount(hard_labels, minlength=6)
                    class_count = class_count + label_count
                
                xyz = f['xyz'][:]
                rgb = f['rgb'][:].astype(np.float)
                elpsv = np.concatenate((f['xyz'][:,2][:,None], f['geof'][:]), axis=1)

                # rescale to [-0.5,0.5]; keep xyz
                elpsv[:,0] /= 100 # (rough guess)
                elpsv[:,1:] -= 0.5
                rgb = rgb/255.0 - 0.5
                
                P = np.concatenate([xyz, rgb, elpsv], axis=1)

                f = h5py.File(pathC + file, 'r')
                numc = len(f['components'].keys())

                with h5py.File(pathP + file, 'w') as hf:
                    hf.create_dataset(name='centroid',data=xyz.mean(0))
                    for c in range(numc):
                        idx = f['components/{:d}'.format(c)][:].flatten()
                        if idx.size > 10000: # trim extra large segments, just for speed-up of loading time
                            ii = random.sample(range(idx.size), k=10000)
                            idx = idx[ii]

                        hf.create_dataset(name='{:d}'.format(c), data=P[idx,...])
    path = '{}/parsed/'.format(SEMA3D_PATH)
    data_file = h5py.File(path+'class_count.h5', 'w')
    data_file.create_dataset('class_count', data=class_count, dtype='int')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs')
    parser.add_argument('--SEMA3D_PATH', default='datasets/semantic3d')
    parser.add_argument('--supervised_partition', default=0, type=int, help = 'wether to use supervized partition features')
    args = parser.parse_args()
    preprocess_pointclouds(args.SEMA3D_PATH)
