import argparse
import time
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from utils.utils_ import log_string, plot_train_val_loss
from utils.utils_ import count_parameters, load_data

from model.model_ import GMAN
from model.train import train
from model.test import test


parser = argparse.ArgumentParser()
parser.add_argument('--time_slot', type=int, default=15, help='a time step is 5 mins')
parser.add_argument('--num_his', type=int, default=20, help='history steps')
parser.add_argument('--num_pred', type=int, default=4, help='prediction steps')
parser.add_argument('--L', type=int, default=3, help='number of STAtt Blocks')
parser.add_argument('--K', type=int, default=8, help='number of attention heads')
parser.add_argument('--d', type=int, default=8, help='dims of each head attention outputs')
parser.add_argument('--train_ratio', type=float, default=1, help='training set [default : 1]')
parser.add_argument('--val_ratio', type=float, default=0.5, help='validation set [default : 0.5]')
parser.add_argument('--test_ratio', type=float, default=0.5, help='testing set [default : 0.5]')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--max_epoch', type=int, default=1, help='epoch to run')
parser.add_argument('--patience', type=int, default=65, help='patience for early stop')
parser.add_argument('--learning_rate', type=float, default=0.01, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=10, help='decay epoch')
parser.add_argument('--train_file', default='./basic/data/train.csv', help='train data CSV file')
parser.add_argument('--val_test_file', default='./basic/data/val_test.csv', help='validation and test data CSV file')
parser.add_argument('--SE_file', default='./basic/data/I-15_NB_SE.txt', help='spatial embedding file')
parser.add_argument('--time_stamp', default='./basic/data/timestamps_new.txt', help='time stamp file')
parser.add_argument('--model_file', default='./basic/data/model_I15NB_0_filled.pt', help='save the model to disk')
parser.add_argument('--log_file', default='./basic/data/log_I15NB_0_filled.txt', help='log file')
parser.add_argument('--cuda_device', type=int, default=0, help='default CUDA device index if GPU is available')

args = parser.parse_args()  # Parsing the command-line arguments

T = 24 * 60 // args.time_slot  # Number of time steps in one day
log = open(args.log_file, 'w')

(trainX, trainTE, trainY, valX, valTE, valY, testX, testTE,
 testY, SE, mean, std) = load_data(args)

print('Data Loaded')

model = GMAN(SE, args, bn_decay=0.1)
loss_criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), args.learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer,
                                      step_size=args.decay_epoch,
                                      gamma=0.9)
parameters = count_parameters(model)

print('Model Compiled')

trainPred, valPred, testPred = test(args, log)
