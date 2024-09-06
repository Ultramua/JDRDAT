from time import gmtime, strftime
from tqdm import tqdm
import torch
import sys
import os
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from build_gen import *
from datasets.dataset_read import dataset_read
from utils import _discrepancy, _ring, _ent, _l2_rec
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

# Append dataset directory to the system path
sys.path.append('./datasets')
print(sys.path)


class Solver():
    def __init__(self, device, args):
        # Initialize solver with device and args
        self.device = device
        self.src_domain_code = np.repeat(np.array([[1, 0]]), args.batch_size, axis=0)
        self.trg_domain_code = np.repeat(np.array([[0, 1]]), args.batch_size, axis=0)
        self.src_domain_code = torch.FloatTensor(self.src_domain_code).to(self.device)
        self.trg_domain_code = torch.FloatTensor(self.trg_domain_code).to(self.device)
        self.source_name = args.source_name
        self.target_name = args.target_name
        self.num_k = args.num_k
        self.batch_size = args.batch_size
        self.lr = args.learning_rate
        self.mi_k = 2  # Mutual Information iterations
        self.delta = 0.01
        self.mi_coeff = 0.0001

        # Load datasets
        print('Loading datasets...')
        self.datasets = dataset_read(
            file_path=args.data_path,
            src_data_name=args.source_file_name,
            tar_data_name=args.target_file_name,
            batch_size=args.batch_size
        )
        print('Dataset loading complete!')

        # Initialize models
        self.G = Generator(self.device, args.num_class, args.input_size, args.sampling_rate, args.num_T,
                           args.out_graph, args.dropout_rate, args.pool, args.pool_step_rate)
        self.FD = Feature_Discriminator(in_features=int(args.out_graph * 0.3), out_features=int(args.out_graph * 0.1))
        self.R = Reconstructor(int(args.out_graph * 0.6), args.out_graph)
        self.MI = Mine(in_features=int(args.out_graph * 0.3), out_features=int(args.out_graph * 0.2))

        # Classifiers and disentanglers
        self.C = nn.ModuleDict({
            'ds': Classifier(int(args.out_graph * 0.3), args.num_class),
            'di': Classifier(int(args.out_graph * 0.3), args.num_class),
            'ci': Classifier(int(args.out_graph * 0.3), 5)
        })
        self.D = nn.ModuleDict({
            'ds': Disentangler(in_feature=args.out_graph, out_feature1=int(args.out_graph * 0.6),
                               out_feature2=int(args.out_graph * 0.3)),
            'di': Disentangler(in_feature=args.out_graph, out_feature1=int(args.out_graph * 0.6),
                               out_feature2=int(args.out_graph * 0.3)),
            'ci': Disentangler(in_feature=args.out_graph, out_feature1=int(args.out_graph * 0.6),
                               out_feature2=int(args.out_graph * 0.3))
        })
        self.modules = nn.ModuleDict({
            'G': self.G, 'FD': self.FD, 'R': self.R, 'MI': self.MI
        })

        # Loss functions
        self.xent_loss = nn.CrossEntropyLoss().cuda(self.device)
        self.adv_loss = nn.BCEWithLogitsLoss().cuda(self.device)
        self.label_smooth_loss = LabelSmoothing().cuda(self.device)

        # Set optimizers and move models to device
        self.set_optimizer(lr=args.learning_rate)
        self.to_device()

    def to_device(self):
        # Move models to the specified device
        for k, v in self.modules.items():
            self.modules[k] = v.cuda(self.device)
        for k, v in self.C.items():
            self.C[k] = v.cuda(self.device)
        for k, v in self.D.items():
            self.D[k] = v.cuda(self.device)

    def set_optimizer(self, lr=0.001):
        # Set up optimizers for each model
        self.opt = {
            'C_ds': optim.Adam(self.C['ds'].parameters(), lr=lr, weight_decay=5e-4),
            'C_di': optim.Adam(self.C['di'].parameters(), lr=lr, weight_decay=5e-4),
            'C_ci': optim.Adam(self.C['ci'].parameters(), lr=lr, weight_decay=5e-4),
            'D_ds': optim.Adam(self.D['ds'].parameters(), lr=lr, weight_decay=5e-4),
            'D_di': optim.Adam(self.D['di'].parameters(), lr=lr, weight_decay=5e-4),
            'D_ci': optim.Adam(self.D['ci'].parameters(), lr=lr, weight_decay=5e-4),
            'G': optim.Adam(self.G.parameters(), lr=lr, weight_decay=5e-4),
            'FD': optim.Adam(self.FD.parameters(), lr=lr, weight_decay=5e-4),
            'R': optim.Adam(self.R.parameters(), lr=lr, weight_decay=5e-4),
            'MI': optim.Adam(self.MI.parameters(), lr=lr, weight_decay=5e-4)
        }

    def reset_grad(self):
        # Reset gradients for all optimizers
        for _, opt in self.opt.items():
            opt.zero_grad()

    def mi_estimator(self, x, y, y_):
        # Estimate mutual information between x, y and shuffled y_
        joint, marginal = self.MI(x, y), self.MI(x, y_)
        return torch.mean(joint) - torch.log(torch.mean(torch.exp(marginal)))

    def group_opt_step(self, opt_keys):
        # Perform optimization step for selected optimizers
        for k in opt_keys:
            self.opt[k].step()
        self.reset_grad()

    def optimize_classifier(self, img_src, img_tar, label_src, emo_label_src, emo_label_tar):
        # Optimization for the classifiers
        feat_src = self.G(img_src)
        feat_tar = self.G(img_tar)
        _loss = dict()
        _loss_emo = dict()

        # Classifier loss for domain-specific tasks
        for key in ['ds', 'di']:
            _loss['class_src_' + key] = self.xent_loss(
                self.C[key](self.D[key](feat_src)), label_src)

        # Emotion classification loss
        for key in ['ci']:
            _loss_emo['class_src_' + key] = self.xent_loss(
                self.C[key](self.D[key](feat_src)), emo_label_src)
            _loss_emo['class_tar_' + key] = self.xent_loss(
                self.C[key](self.D[key](feat_tar)), emo_label_tar)

        # Total loss summation
        _sum_loss = sum([l for _, l in _loss.items()]) + 0.1 * sum([l for _, l in _loss_emo.items()])
        _sum_loss.backward()

        # Perform optimization steps
        self.group_opt_step(['G', 'C_ds', 'C_di', 'C_ci', 'D_ds', 'D_di', 'D_ci'])
        return _loss

    def discrepancy_minimizer(self, img_src, img_trg, label_src):
        # Minimizes discrepancy between classifiers on source and target domains
        _loss = dict()

        # Source domain loss
        _loss['ds_src'] = self.xent_loss(self.C['ds'](self.D['ds'](self.G(img_src))), label_src)
        _loss['di_src'] = self.xent_loss(self.C['di'](self.D['di'](self.G(img_src))), label_src)

        # Target domain discrepancy
        _loss['discrepancy_ds_di_trg'] = _discrepancy(
            self.C['ds'](self.D['ds'](self.G(img_trg))),
            self.C['di'](self.D['di'](self.G(img_trg)))
        )

        # Backpropagation
        _sum_loss = sum([l for _, l in _loss.items()])
        _sum_loss.backward()

        # Optimize corresponding networks
        self.group_opt_step(['D_ds', 'D_di', 'C_ds', 'C_di'])
        return _loss

    def ring_loss_minimizer(self, img_src, img_trg):
        # Minimize ring loss to regularize feature space
        data = torch.cat((img_src, img_trg), 0)
        feat = self.G(data)
        ring_loss = _ring(feat)
        ring_loss.backward()
        self.group_opt_step(['G'])
        return ring_loss

    def mutual_information_minimizer(self, img_src, img_trg):
        # Minimize mutual information between different feature representations
        for _ in range(self.mi_k):
            ds_src, ds_trg = self.D['ds'](self.G(img_src)), self.D['ds'](self.G(img_trg))
            di_src, di_trg = self.D['di'](self.G(img_src)), self.D['di'](self.G(img_trg))
            ci_src, ci_trg = self.D['ci'](self.G(img_src)), self.D['ci'](self.G(img_trg))

            # Shuffle data for marginal estimation
            ci_src_shuffle = torch.index_select(ci_src, 0, torch.randperm(ci_src.shape[0]).to(self.device))
            ci_trg_shuffle = torch.index_select(ci_trg, 0, torch.randperm(ci_trg.shape[0]).to(self.device))
            ds_src_shuffle = torch.index_select(ds_src, 0, torch.randperm(ds_src.shape[0]).to(self.device))
            ds_trg_shuffle = torch.index_select(ds_trg, 0, torch.randperm(ds_trg.shape[0]).to(self.device))

            MI_di_ds_src = self.mi_estimator(di_src, ds_src, ds_src_shuffle)
            MI_di_ds_trg = self.mi_estimator(di_trg, ds_trg, ds_trg_shuffle)
            MI_di_ci_src = self.mi_estimator(di_src, ci_src, ci_src_shuffle)
            MI_di_ci_trg = self.mi_estimator(di_trg, ci_trg, ci_trg_shuffle)

            MI = 0.25 * (MI_di_ds_src + MI_di_ds_trg + MI_di_ci_src + MI_di_ci_trg) * self.mi_coeff
            MI.backward()

            # Optimize networks
            self.group_opt_step(['D_ds', 'D_di', 'D_ci', 'MI'])

    def adversarial_alignment(self, img_src, img_trg):
        # Adversarial alignment of source and target domains
        for _ in range(self.num_k):
            src_domain_pred = self.FD(self.D['di'](self.G(img_src)))
            tgt_domain_pred = self.FD(self.D['di'](self.G(img_trg)))
            df_loss_src = self.adv_loss(src_domain_pred, self.src_domain_code)
            df_loss_trg = self.adv_loss(tgt_domain_pred, self.trg_domain_code)
            alignment_loss1 = 0.01 * (df_loss_src + df_loss_trg)
            alignment_loss1.backward()
            self.group_opt_step(['FD', 'G'])

        for _ in range(self.num_k):
            tgt_domain_pred = self.FD(self.D['di'](self.G(img_trg)))
            df_loss_trg = self.adv_loss(tgt_domain_pred, 1 - self.trg_domain_code)
            alignment_loss2 = 0.01 * df_loss_trg
            alignment_loss2.backward()
            self.group_opt_step(['D_di', 'G'])

        for _ in range(self.num_k):
            loss_dis = _discrepancy(
                self.C['ds'](self.D['ds'](self.G(img_trg))),
                self.C['di'](self.D['di'](self.G(img_trg))))
            loss_dis.backward()
            self.group_opt_step(['G', 'D_di', 'D_ds'])

        return alignment_loss1, alignment_loss2, loss_dis

    def optimize_rec(self, img_src, img_trg):
        # Reconstruction loss optimization
        _feat_src = self.G(img_src)
        _feat_trg = self.G(img_trg)

        feat_src, feat_trg = dict(), dict()
        rec_src, rec_trg = dict(), dict()
        for k in ['ds', 'di', 'ci']:
            feat_src[k] = self.D[k](_feat_src)
            feat_trg[k] = self.D[k](_feat_trg)

        recon_loss = None
        rec_loss_src, rec_loss_trg = dict(), dict()
        for k1, k2 in [('ds', 'ci'), ('di', 'ci')]:
            k = '%s_%s' % (k1, k2)
            rec_src[k] = self.R(torch.cat([feat_src[k1], feat_src[k2]], 1))
            rec_trg[k] = self.R(torch.cat([feat_trg[k1], feat_trg[k2]], 1))
            rec_loss_src[k] = _l2_rec(rec_src[k], _feat_src)
            rec_loss_trg[k] = _l2_rec(rec_trg[k], _feat_trg)

            recon_loss = rec_loss_src[k] + rec_loss_trg[k] if recon_loss is None else recon_loss + rec_loss_src[k] + rec_loss_trg[k]

        recon_loss = (recon_loss / 4) * self.delta
        recon_loss.backward()
        self.group_opt_step(['D_di', 'D_ci', 'D_ds', 'R'])
        return rec_loss_src, rec_loss_trg

    def train_epoch(self, epoch, record_file=None):
        # Training for one epoch
        for k in self.modules.keys():
            self.modules[k].train()
        for k in self.C.keys():
            self.C[k].train()
        for k in self.D.keys():
            self.D[k].train()

        total_batches = 220
        for batch_idx, data in enumerate(self.datasets):
            if batch_idx > total_batches:
                return batch_idx

            data_trg = data['T'].to(self.device)
            data_src = data['S'].to(self.device)
            label_src = data['S_label'].long().to(self.device)
            emo_label_src = data['S_emo_label'].long().to(self.device)
            emo_label_tar = data['T_emo_label'].long().to(self.device)

            if data_src.size()[0] < self.batch_size or data_trg.size()[0] < self.batch_size:
                break

            self.reset_grad()

            # Optimize classifier, alignment, and minimize losses
            class_loss = self.optimize_classifier(data_src, data_trg, label_src, emo_label_src, emo_label_tar)
            confusion_loss = self.class_confusion(data_src, data_trg)
            (alignment_loss1, alignment_loss2, discrepancy_loss) = self.adversarial_alignment(data_src, data_trg)
            self.mutual_information_minimizer(data_src, data_trg)
            ring_loss = self.ring_loss_minimizer(data_src, data_trg)

        return batch_idx

    def test(self, epoch):
        # Model evaluation and testing
        self.G.eval()
        self.D['di'].eval()
        self.D['ds'].eval()
        self.C['di'].eval()
        self.C['ds'].eval()

        test_loss = 0
        size = 0
        emo_val = []
        act_val = []
        correct1, correct2, correct3, correct4 = 0, 0, 0, 0
        pred_di_val = []
        pred_ds_val = []
        pred_emo_val = []
        pred_total_val = []
        vl = Averager()

        with torch.no_grad():
            for batch_idx, data in enumerate(self.datasets):
                data, label, emo_label = data['T'], data['T_label'].long(), data['T_emo_label'].long()
                data, label, emo_label = data.to(self.device), label.to(self.device), emo_label.to(self.device)

                feat = self.G(data)
                out1 = self.C['di'](self.D['di'](feat))
                out2 = self.C['ds'](self.D['ds'](feat))
                out3 = self.C['ci'](self.D['ci'](feat))

                test_loss += F.nll_loss(out1, label).item()
                test_loss += F.nll_loss(out3, emo_label).item()

                out_ensemble = out1 + out2
                pred_di = out1.data.max(1)[1]
                pred_ds = out2.data.max(1)[1]
                pred_emo = out3.data.max(1)[1]
                pred_total = out_ensemble.data.max(1)[1]

                pred_di_val.extend(pred_di.data.tolist())
                pred_ds_val.extend(pred_ds.data.tolist())
                pred_emo_val.extend(pred_emo.data.tolist())
                pred_total_val.extend(pred_total.data.tolist())
                act_val.extend(label.data.tolist())
                emo_val.extend(emo_label.data.tolist())

                vl.add(test_loss)

                k = label.data.size()[0]
                correct1 += pred_di.eq(label.data).cpu().sum()
                correct2 += pred_ds.eq(label.data).cpu().sum()
                correct3 += pred_emo.eq(emo_label.data).cpu().sum()
                correct4 += pred_total.eq(label.data).cpu().sum()
                size += k

        test_loss = test_loss / size
        di_f1 = f1_score(act_val, pred_di_val, average='macro')
        ds_f1 = f1_score(act_val, pred_ds_val, average='macro')
        total_f1 = f1_score(act_val, pred_total_val, average='macro')
        precision = precision_score(act_val, pred_di_val, average='macro', zero_division=0)
        recall = recall_score(act_val, pred_di_val, average='macro')
        cm = confusion_matrix(act_val, pred_di_val)

        acc_di = correct1 / size
        acc_ds = correct2 / size
        acc_emo = correct3 / size
        acc_total = correct4 / size

        print(
            f'\nepoch:{epoch},Test set: Average loss: {test_loss:.4f}, Accuracy C1: {correct1}/{size} ({acc_di:.4f}) '
            f'Accuracy C2: {correct2}/{size} ({acc_ds:.4f}) emo_Accuracy C3: {correct3}/{size} ({acc_emo:.4f}) '
            f'Accuracy Ensemble: {correct4}/{size} ({acc_total:.4f}) di_f1:{di_f1:.4f} ,ds_f1:{ds_f1:.4f},'
            f'total_f1:{total_f1:.4f}\n')
        return acc_di, acc_ds, acc_emo, acc_total, di_f1, ds_f1, total_f1, precision, recall, cm

    def uda_class_alignment_loss(self, x_source, x_target, pseudo_classes, classes):
        # UDA class alignment loss to align source and target samples
        x_source_copy = x_source
        x_target_copy = x_target

        pseudo_classes_target_copy = torch.argmax(pseudo_classes, dim=1)
        classes_source_copy = classes

        # Compute prototypes for each class as the mean of the features
        source_dict = dict(zip(x_source_copy, classes_source_copy))
        target_dict = dict(zip(x_target_copy, pseudo_classes_target_copy))

        final_source_dict = {key: torch.mean(torch.stack([k for k, v in source_dict.items() if v == key]), dim=0)
                             for key in set(source_dict.values())}
        final_target_dict = {key: torch.mean(torch.stack([k for k, v in target_dict.items() if v == key]), dim=0)
                             for key in set(target_dict.values())}

        # Compute squared euclidean distances between prototypes of the same classes
        sum_dists = sum(((final_source_dict[k] - final_target_dict[k]) ** 2).sum() for k in final_source_dict if k in final_target_dict)
        return sum_dists
