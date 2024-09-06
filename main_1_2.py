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
import os.path as osp

# Argument parser for configuring the training settings
parser = argparse.ArgumentParser(description='PyTorch MCD Implementation')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='Input batch size for training')
parser.add_argument('--dataset', type=str, default='SEED_V', help='Dataset name')
parser.add_argument('--data_path', type=str,
                    default="/home/lhg/processed_data/second_work/Tsinghua_rsvp_data/full_band/all_data/",
                    help='Path to the dataset')
parser.add_argument('--dropout_rate', type=float, default=0.25, metavar='LR', help='Dropout rate')
parser.add_argument('--input_size', type=tuple, default=(1, 62, 500), help='Input size')
parser.add_argument('--learning_rate', type=float, default=0.0001, metavar='LR', help='Learning rate')
parser.add_argument('--max_epoch', type=int, default=100, metavar='N', help='Number of epochs')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disable CUDA training')
parser.add_argument('--num_T', type=int, default=16, metavar='N', help='Time convolution hyperparameter')
parser.add_argument('--num_k', type=int, default=4, metavar='N', help='Generator update hyperparameter')
parser.add_argument('--num_class', type=int, default=14, metavar='N', help='Number of classes for classification')
parser.add_argument('--optimizer', type=str, default='adam', metavar='N', help='Optimizer choice')
parser.add_argument('--resume_epoch', type=int, default=100, metavar='N', help='Epoch to resume training')
parser.add_argument('--save_epoch', type=int, default=10, metavar='N', help='Epoch interval for saving models')
parser.add_argument('--sampling_rate', type=int, default=500, metavar='N', help='Sampling rate for data')
parser.add_argument('--seed', type=int, default=2023, metavar='S', help='Random seed')
parser.add_argument('--source_name', type=str, default='session1', metavar='N', help='Source session name')
parser.add_argument('--source_file_name', type=str, default='session1.hdf', help='Source file name')
parser.add_argument('--target_name', type=str, default='session2', metavar='N', help='Target session name')
parser.add_argument('--target_file_name', type=str, default='session2.hdf', help='Target file name')
parser.add_argument('--out_graph', type=int, default=256, help='Graph output size')
parser.add_argument('--pool', type=int, default=20, help='Pooling size')
parser.add_argument('--pool_step_rate', type=float, default=0.25, help='Pooling step rate')
parser.add_argument('--exp_name', type=str, default='test', metavar='N', help='Experiment name')
parser.add_argument('--gpu', type=int, default=3, metavar='S', help='GPU index')
parser.add_argument('--use_cuda', action='store_true', default=True, help='Use CUDA if available')

# Parse the arguments
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.cuda.set_device(args.gpu)

# Print the parsed arguments
print(args)


def main():
    # Initialize logging
    log_path = './log'
    ensure_path(log_path)
    log_path = os.path.join(log_path, 'RSVP_1_2.txt')
    sys.stdout = Tee(log_path)  # Redirect console output to a file
    seed_all(2023)

    file_name = 'RSVP_1_2'
    device = torch.device("cuda:3" if args.use_cuda else "cpu")
    solver = Solver(device=device, args=args)

    count = 0
    max_di_acc = 0.0
    max_di_f1 = 0.0
    max_ds_acc = 0.0
    max_ds_f1 = 0.0

    # Function to save the model states
    def save_model(name):
        save_path = '/home/lhg/model_weight/second_work/RSVEP/s1_s2_has_ring_no_emo'
        ensure_path(save_path)
        torch.save(solver.G.state_dict(), osp.join(save_path, '{}_G.pth'.format(name)))
        torch.save(solver.D.state_dict(), osp.join(save_path, '{}_D.pth'.format(name)))
        torch.save(solver.C.state_dict(), osp.join(save_path, '{}_C.pth'.format(name)))
        torch.save(solver.FD.state_dict(), osp.join(save_path, '{}_FD.pth'.format(name)))
        torch.save(solver.MI.state_dict(), osp.join(save_path, '{}_MI.pth'.format(name)))
        torch.save(solver.R.state_dict(), osp.join(save_path, '{}_R.pth'.format(name)))

    # Main training loop
    for epoch in range(args.max_epoch):
        num = solver.train_epoch(epoch)
        count += num

        # Test the model and update the metrics
        if epoch % 1 == 0:
            acc_di, acc_ds, acc_emo, acc_total, di_f1, ds_f1, total_f1, precision, recall, cm = solver.test(epoch)

            if acc_di >= max_di_acc:
                max_di_acc = acc_di
                max_di_f1 = di_f1
                max_ds_acc = acc_ds
                max_ds_f1 = ds_f1

                print(
                    f'max_di_acc: {max_di_acc}, max_di_f1: {max_di_f1}, max_ds_acc: {max_ds_acc}, max_ds_f1: {max_ds_f1}')

                # Convert confusion matrix to percentages
                cm_percent = cm / cm.sum(axis=1)[:, None] * 100

                # Create a heatmap for the confusion matrix
                plt.close()
                plt.figure(figsize=(30, 30))
                sns.heatmap(cm_percent, annot=True, fmt='.2f', cmap='Blues')
                plt.title('Confusion Matrix')
                plt.ylabel('Actual')
                plt.xlabel('Predicted')

                # Save confusion matrix as an image
                plt.savefig(file_name + str(args.num_class) + '_confusion_matrix.png')

                # Save confusion matrix as a CSV file
                with open(file_name + str(args.num_class) + '_confusion_matrix.csv', 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([''] + [f'Predicted {i}' for i in range(len(cm))])
                    for i, row in enumerate(cm):
                        writer.writerow([f'Actual {i}'] + list(row))


if __name__ == '__main__':
    main()
