import torch
from utils import *
from Tee import *
import h5py
from Solver import Solver
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys

parser = argparse.ArgumentParser(description='PyTorch MCD Implementation')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 2)')
parser.add_argument('--dataset', type=str, default='RSVP')
parser.add_argument('--data_path', type=str,
                    default="/home/lhg/processed_data/second_work/Tsinghua_rsvp_data/full_band/all_data/")
parser.add_argument('--dropout_rate', type=float, default=0.25, metavar='LR')
parser.add_argument('--input_size', type=tuple, default=(1, 62, 500))
parser.add_argument('--learning_rate', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--max_epoch', type=int, default=100, metavar='N',
                    help='how many epochs')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--num_T', type=int, default=16, metavar='N',
                    help='hyper paremeter for time conv')
parser.add_argument('--num_k', type=int, default=4, metavar='N',
                    help='hyper paremeter for generator update')
parser.add_argument('--num_class', type=int, default=14, metavar='N',
                    help='number for classification')
parser.add_argument('--optimizer', type=str, default='adam', metavar='N', help='which optimizer')
parser.add_argument('--resume_epoch', type=int, default=100, metavar='N',
                    help='epoch to resume')
parser.add_argument('--save_epoch', type=int, default=10, metavar='N',
                    help='when to restore the model')
parser.add_argument('--sampling_rate', type=int, default=500, metavar='N',
                    help='length for data')
parser.add_argument('--seed', type=int, default=2023, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--source_name', type=str, default='session1', metavar='N',
                    help='source session')
parser.add_argument('--source_file_name', type=str, default='session1.hdf')
parser.add_argument('--target_name', type=str, default='session2', metavar='N', help='target session')
parser.add_argument('--target_file_name', type=str, default='session2.hdf')
parser.add_argument('--out_graph', type=int, default=512)
parser.add_argument('--pool', type=int, default=20)
parser.add_argument('--pool_step_rate', type=float, default=0.25)
parser.add_argument('--exp_name', type=str, default='test', metavar='N')
parser.add_argument('--gpu', type=int, default=3, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--use_cuda', action='store_true', default=True,
                    help='Use cuda or not')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.cuda.set_device(args.gpu)
print(args)


def main():
    log_path = './log'
    ensure_path(log_path)
    log_path = os.path.join(log_path, 'RSVP_1_2.txt')
    sys.stdout = Tee(log_path)
    seed_all(2023)
    file_name='RSVP_1_2'
    # set_gpu('1')
    device = torch.device("cuda:3" if args.use_cuda else "cpu")
    solver = Solver(device=device, args=args)
    count = 0
    max_di_acc = 0.0
    max_di_f1 = 0.0

    max_ds_acc = 0.0
    max_ds_f1 = 0.0
    def save_model(name):
        save_path = '/home/lhg/model_weight/second_work/RSVEP/s1_s2_has_ring_no_emo'
        ensure_path(save_path)
        # previous_model = osp.join(save_path, '{}.pth'.format(name))
        # if os.path.exists(previous_model):
        #     os.remove(previous_model)
        torch.save(solver.G.state_dict(), osp.join(save_path, '{}_G.pth'.format(name)))
        torch.save(solver.D.state_dict(), osp.join(save_path, '{}_D.pth'.format(name)))
        torch.save(solver.C.state_dict(), osp.join(save_path, '{}_C.pth'.format(name)))
        torch.save(solver.FD.state_dict(), osp.join(save_path, '{}_FD.pth'.format(name)))
        torch.save(solver.MI.state_dict(), osp.join(save_path, '{}_MI.pth'.format(name)))
        torch.save(solver.R.state_dict(), osp.join(save_path, '{}_R.pth'.format(name)))

    for epoch in range(args.max_epoch):
        num = solver.train_epoch(epoch)
        count += num
        if epoch % 1 == 0:
            acc_di, acc_ds, acc_emo, acc_total ,di_f1,ds_f1,total_f1,precision,recall,cm = solver.test(epoch)
            if acc_di >= max_di_acc:
                max_di_acc = acc_di
                max_di_f1 = di_f1

                max_ds_acc = acc_ds
                max_ds_f1 = ds_f1
                print('max_di_acc is:{},max_di_f1 is :{},max_ds_acc is:{},max_ds_f1 is:{}'.format(max_di_acc, max_di_f1,
                                                                                                  max_ds_acc,
                                                                                                  max_ds_f1))

                cm_percent = cm / cm.sum(axis=1)[:, None] * 100


                plt.close()
                plt.figure(figsize=(30, 30))
                sns.heatmap(cm_percent, annot=True, fmt='.2f', cmap='Blues')
                plt.title('Confusion Matrix')
                plt.ylabel('Actual')
                plt.xlabel('Predicted')


                plt.savefig(file_name + str(args.num_class) + '_confusion_matrix.png')


                # plt.show()
                with open(file_name + str(args.num_class) + '_confusion_matrix.csv', 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([''] + [f'Predicted {i}' for i in range(len(cm))])
                    for i, row in enumerate(cm):
                        writer.writerow([f'Actual {i}'] + list(row))

            # save_model('max_acc_' + str(np.round(max_di_acc.numpy(), 4)))
        # if count >= 20000:
        #     break


if __name__ == '__main__':
    main()
