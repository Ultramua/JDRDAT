import torch.utils.data
from builtins import object
from datasets.eeg_dataset import eegDataset


class PairedData(object):
    def __init__(self, data_loader_A, data_loader_B, max_dataset_size):
        self.data_loader_A = data_loader_A
        self.data_loader_B = data_loader_B
        self.stop_A = False
        self.stop_B = False
        self.max_dataset_size = max_dataset_size

    def __iter__(self):
        self.stop_A = False
        self.stop_B = False
        self.data_loader_A_iter = iter(self.data_loader_A)
        self.data_loader_B_iter = iter(self.data_loader_B)
        self.iter = 0
        return self

    def __next__(self):
        A, A_paths, A_emo = None, None, None
        B, B_paths, B_emo = None, None, None
        try:
            A, A_paths, A_emo = next(self.data_loader_A_iter)
        except StopIteration:
            if A is None or A_paths is None or A_emo is None:
                self.stop_A = True
                self.data_loader_A_iter = iter(self.data_loader_A)
                A, A_paths, A_emo = next(self.data_loader_A_iter)

        try:
            B, B_paths, B_emo = next(self.data_loader_B_iter)
        except StopIteration:
            if B is None or B_paths is None or B_emo is None:
                self.stop_B = True
                self.data_loader_B_iter = iter(self.data_loader_B)
                B, B_paths, B_emo = next(self.data_loader_B_iter)

        if (self.stop_A and self.stop_B) or self.iter > self.max_dataset_size:
            self.stop_A = False
            self.stop_B = False
            raise StopIteration()
        else:
            self.iter += 1
            return {'S': A, 'S_label': A_paths, 'S_emo_label': A_emo,
                    'T': B, 'T_label': B_paths, 'T_emo_label': B_emo}


class UnalignedDataLoader():
    def initialize(self, source, target, batch_size1, batch_size2):
        dataset_source = eegDataset(source['data'], source['labels'], source['emo_labels'])
        dataset_target = eegDataset(target['data'], target['labels'], target['emo_labels'])
        data_loader_s = torch.utils.data.DataLoader(
            dataset_source,
            batch_size=batch_size1,
            shuffle=True,
            generator=torch.Generator(device='cpu'),
            num_workers=0)

        data_loader_t = torch.utils.data.DataLoader(
            dataset_target,
            batch_size=batch_size2,
            shuffle=True,
            generator=torch.Generator(device='cpu'),
            num_workers=0)
        self.dataset_s = dataset_source
        self.dataset_t = dataset_target
        # self.max_batch_size=min(len(source['labels'])//batch_size1,len(target['labels'])//batch_size2)
        self.paired_data = PairedData(data_loader_s, data_loader_t,
                                      float("inf"))

    def name(self):
        return 'UnalignedDataLoader'

    def load_data(self):
        return self.paired_data

    def __len__(self):
        return min(max(len(self.dataset_s), len(self.dataset_t)), float("inf"))
