import sys
import numpy as np

sys.path.append('../loader')

import os
import h5py
from utils import *
from .unaligned_data_loader import UnalignedDataLoader


def dataset_read(file_path, src_data_name, tar_data_name, batch_size):
    # load data

    S = {}
    S_test = {}
    T = {}
    T_test = {}
    file_path = file_path
    src_data_name = src_data_name
    src_data_path = os.path.join(file_path, src_data_name)
    dataset = h5py.File(src_data_path, 'r')
    src_data = np.array(dataset['data'])
    if len(np.shape(src_data)) < 4:
        src_data = np.expand_dims(src_data, axis=1)
    src_label = np.array(dataset['label'])
    src_emo_label = np.array(dataset['task_label'])

    print('>>> total -->Src Data:{} Src Label:{} Src emo_Label:{}'.format(src_data.shape, src_label.shape,
                                                                          src_emo_label.shape))

    tar_data_name = tar_data_name
    tar_data_path = os.path.join(file_path, tar_data_name)
    dataset = h5py.File(tar_data_path, 'r')
    tar_data = np.array(dataset['data'])
    if len(np.shape(tar_data)) < 4:
        tar_data = np.expand_dims(tar_data, axis=1)
    tar_label = np.array(dataset['label'])
    tar_emo_label = np.array(dataset['task_label'])


    print('>>> total -->Test Data:{} Test Label:{} Test emo_Label:{} '.format(tar_data.shape, tar_label.shape,
                                                                              tar_emo_label.shape))

    src_data, tar_data = normalize_v3(src_data, tar_data)
    src_data = torch.from_numpy(src_data).float()
    src_label = torch.from_numpy(src_label).long()
    src_emo_label = torch.from_numpy(src_emo_label).long()
    tar_data = torch.from_numpy(tar_data).float()
    tar_label = torch.from_numpy(tar_label).long()
    tar_emo_label = torch.from_numpy(tar_emo_label).long()
    print('Data and label prepared!')
    print('>>> Test Data:{} Test Label:{} Test emo_Label:{} '.format(tar_data.shape, tar_label.shape,
                                                                     tar_emo_label.shape))
    print(
        '>>> Train Data:{}Train Label:{} Src emo_Label:{}'.format(src_data.shape, src_label.shape, src_emo_label.shape))
    print('----------------------')

    S['data'] = src_data
    S['labels'] = src_label
    S['emo_labels'] = src_emo_label

    T['data'] = tar_data
    T['labels'] = tar_label
    T['emo_labels'] = tar_emo_label

    train_loader = UnalignedDataLoader()
    train_loader.initialize(S, T, batch_size2=batch_size, batch_size1=batch_size)
    dataset = train_loader.load_data()
    return dataset
