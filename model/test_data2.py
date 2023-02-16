import argparse
import time
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from utils.utils_ import log_string, plot_train_val_loss
from utils.utils_ import count_parameters, load_data

from model.model_ import GMAN
from model.train import train
from model.test_data import test

parser = argparse.ArgumentParser()
parser.add_argument('--time_slot', type=int, default=5,
                    help='a time step is 5 mins')
parser.add_argument('--num_his', type=int, default=12,
                    help='history steps')
parser.add_argument('--num_pred', type=int, default=12,
                    help='prediction steps')
parser.add_argument('--L', type=int, default=5,
                    help='number of STAtt Blocks')
parser.add_argument('--K', type=int, default=8,
                    help='number of attention heads')
parser.add_argument('--d', type=int, default=8,
                    help='dims of each head attention outputs')
parser.add_argument('--train_ratio', type=float, default=0.05,
                    help='training set [default : 0.7]')
parser.add_argument('--val_ratio', type=float, default=0.05,
                    help='validation set [default : 0.1]')
parser.add_argument('--test_ratio', type=float, default=0.9,
                    help='testing set [default : 0.2]')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size')
parser.add_argument('--max_epoch', type=int, default=5,
                    help='epoch to run')
parser.add_argument('--patience', type=int, default=10,
                    help='patience for early stop')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=10,
                    help='decay epoch')
parser.add_argument('--traffic_file', default='/home/zahid/test-run/GMAN-PyTorch-master/GMAN-PyTorch-master/data/speed2019modified.h5',
                    help='traffic file')
parser.add_argument('--SE_file', default='/home/zahid/test-run/GMAN-PyTorch-master/GMAN-PyTorch-master/data/SE(LAS-28).txt',
                    help='spatial embedding file')
parser.add_argument('--model_file', default="/home/zahid/test-run/GMAN-PyTorch-master/GMAN-PyTorch-master/data/GMAN_speed2018.pkl",
                    help='save the model to disk')
parser.add_argument('--log_file', default='/home/zahid/test-run/GMAN-PyTorch-master/GMAN-PyTorch-master/data/log',
                    help='log file')
args = parser.parse_args()
log = open(args.log_file, 'w')
log_string(log, str(args)[10: -1])
T = 24 * 60 // args.time_slot  # Number of time steps in one day
# load data
log_string(log, 'loading data...')
(trainX, trainTE, trainY, valX, valTE, valY, testX, testTE,
 testY, SE, mean, std) = load_data(args)
log_string(log, f'trainX: {trainX.shape}\t\t trainY: {trainY.shape}')
log_string(log, f'valX:   {valX.shape}\t\tvalY:   {valY.shape}')
log_string(log, f'testX:   {testX.shape}\t\ttestY:   {testY.shape}')
log_string(log, f'mean:   {mean:.4f}\t\tstd:   {std:.4f}')
log_string(log, 'data loaded!')
del trainX, trainTE, valX, valTE, testX, testTE, mean, std


def test_run():
    # start = time.time()
    # loss_train, loss_val = train(model, args, log, loss_criterion, optimizer, scheduler)
    # plot_train_val_loss(loss_train, loss_val, 'figure/train_val_loss.png')
    testPred = test(args, log)
    # end = time.time()
    # log_string(log, 'total time: %.1fmin' % ((end - start) / 60))
    # log.close()
    # trainPred_ = trainPred.numpy().reshape(-1, trainY.shape[-1])
    # trainY_ = trainY.numpy().reshape(-1, trainY.shape[-1])
    # valPred_ = valPred.numpy().reshape(-1, valY.shape[-1])
    # valY_ = valY.numpy().reshape(-1, valY.shape[-1])
    testPred_ = testPred.numpy().reshape(-1, testY.shape[-1])
    testY_ = testY.numpy().reshape(-1, testY.shape[-1])


test_run()
