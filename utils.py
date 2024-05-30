import time
import os
import numpy as np
import random
import datetime
from datasets.eeg_dataset import *
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import os.path as osp
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
print('use_cuda:', use_cuda)
device = torch.device('cuda:0' if use_cuda else 'cpu')


def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)


def set_up(args):
    set_gpu(args.gpu)
    ensure_path(args.save_path)
    torch.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True


def set_gpu(x):
    torch.cuda.device_count()
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    torch.set_num_threads(1)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)


def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)


def get_dataloader(data, label, batch_size):
    # load the data  ; generator=torch.Generator(device=device),
    dataset = eegDataset(data, label)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, pin_memory=False,
                        generator=torch.Generator(device=device), drop_last=True)
    return loader


def get_metrics(y_pred, y_true, classes=None):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    if classes is not None:
        cm = confusion_matrix(y_true, y_pred, labels=classes)
    else:
        cm = confusion_matrix(y_true, y_pred)
    return acc, f1, cm


def record_init(args):
    result_path = osp.join(args.save_path, 'result')
    ensure_path(result_path)
    text_file = osp.join(result_path,
                         "results_{}.txt".format(args.dataset))
    file = open(text_file, 'a')
    file.write("\n" + str(datetime.datetime.now()) +
               "\nTrain:Parameter setting for " + str(args.model) + ' on ' + str(args.dataset) +
               "\n1)number_class:" + str(args.num_class) +
               "\n2)random_seed:" + str(args.random_seed) +
               "\n3)learning_rate:" + str(args.learning_rate) +
               "\n4)pool:" + str(args.pool) +
               "\n5)num_epochs:" + str(args.max_epoch) +
               "\n6)batch_size:" + str(args.batch_size) +
               "\n7)dropout:" + str(args.dropout) +
               "\n8)hidden1_node:" + str(args.hidden1) + "hidden2_node:" + str(args.hidden2) +
               "\n9)input_shape:" + str(args.input_shape) +
               "\n10)train setting:" + str(args.train_session) + str(args.train_emotion) +
               "\n11)test setting:" + str(args.test_session) + str(args.test_emotion) +
               "\n12)T:" + str(args.T) + '\n')

    file.close()


def log2txt(content, args):
    """
    this function log the content to results.txt
    :param content: string, the content to log
    """
    result_path = osp.join(args.save_path, 'result')
    ensure_path(result_path)
    text_file = osp.join(result_path,
                         "results_{}.txt".format(args.dataset))
    file = open(text_file, 'a')
    file.write(str(content) + '\n')
    file.close()


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


class LabelSmoothing(nn.Module):
    """NLL loss with label smoothing.
       refer to: https://github.com/NVIDIA/DeepLearningExamples/blob/8d8b21a933fff3defb692e0527fca15532da5dc6/PyTorch/Classification/ConvNets/image_classification/smoothing.py#L18
    """

    def __init__(self, smoothing=0.0):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)


def resampling(data):
    mean = np.mean(data, axis=-1, keepdims=True)
    data = data - mean
    return data


def halved_data(train_data, train_label, test_data, test_label):
    train_data = train_data[::5]
    train_label = train_label[::5]
    test_data = test_data[::5]
    test_label = test_label[::5]
    return train_data, train_label, test_data, test_label


def normalize(input):

    for channel in range(input.shape[2]):
        input_mean = np.mean(input[:, :, channel, :])
        input_std = np.std(input[:, :, channel, :])
        input[:, :, channel, :] = (input[:, :, channel, :] - input_mean) / input_std
    return input


def normalize_v3(train, test):
    """
    this function do standard normalization for EEG channel by channel
    :param train: training data (sample, 1, chan, datapoint)
    :param test: testing data (sample, 1, chan, datapoint)
    :return: normalized training and testing data
    """


    for channel in range(train.shape[2]):
        train_mean = np.mean(train[:, :, channel, :])
        test_mean = np.mean(test[:, :, channel, :])
        train_std = np.std(train[:, :, channel, :])
        test_std = np.std(test[:, :, channel, :])
        train[:, :, channel, :] = (train[:, :, channel, :] - train_mean) / train_std
        test[:, :, channel, :] = (test[:, :, channel, :] - test_mean) / test_std
    return train, test


def _l2_rec(src, trg):
    return torch.sum((src - trg) ** 2) / (src.shape[0] * src.shape[1])


def _ent(out):
    return torch.mean(torch.log(F.softmax(out + 1e-6, dim=-1)))
#  return  -torch.mean(torch.log(F.softmax(out + 1e-6, dim=-1)))


def _discrepancy(out1, out2):
    return torch.mean(torch.abs(F.softmax(out1, dim=-1) - F.softmax(out2, dim=-1)))


def _ring(feat, type='geman'):
    x = feat.pow(2).sum(dim=1).pow(0.5)
    radius = x.mean()
    radius = radius.expand_as(x)
    # print(radius)
    if type == 'geman':
        l2_loss = (x - radius).pow(2).sum(dim=0) / (x.shape[0] * 0.5)
        return l2_loss
    else:
        raise NotImplementedError("Only 'geman' is implemented")


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)


def dense_to_one_hot(labels_dense):
    """Convert class labels from scalars to one-hot vectors."""
    labels_one_hot = np.zeros((len(labels_dense),))
    labels_dense = list(labels_dense)
    for i, t in enumerate(labels_dense):
        if t == 10:
            t = 0
            labels_one_hot[i] = t
        else:
            labels_one_hot[i] = t
    return labels_one_hot
