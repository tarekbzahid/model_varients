
#######################################################################################################################
# this version of the gman will be trained on the masked data
# the train dataset is made with two types of data, one with missing values and the other with imputed values
# the historical data contains missing values filled with 0 or -1 hence masking the data
# the prediction data contains the imputed values - giving us as close as possible to the real data
# the model therefore will hopefully learn to predict the imputed values from the masked data




import argparse  # Importing the argparse module to parse command-line arguments
import time  # Importing the time module to measure execution time
import torch  # Importing PyTorch
import torch.optim as optim  # Importing the optimization module from PyTorch
import torch.nn as nn  # Importing the neural network module from PyTorch
import numpy as np  # Importing NumPy for numerical computations
import matplotlib.pyplot as plt  # Importing Matplotlib for plotting

# Importing utility functions and classes from custom modules
from utils.utils_ import log_string, plot_train_val_loss
from utils.utils_ import count_parameters, load_data

# Importing model architecture and training/testing functions from custom modules
from model.model_ import GMAN
from model.train import train
from model.test import test

def main():
    parser = argparse.ArgumentParser(description='Train a model on masked and imputed data')  
    # Adding command-line arguments
    parser.add_argument('--time_slot', type=int, default=15, help='a time step is 5 mins')
    parser.add_argument('--num_his', type=int, default=20, help='history steps')
    parser.add_argument('--num_pred', type=int, default=4, help='prediction steps')
    parser.add_argument('--L', type=int, default=3, help='number of STAtt Blocks')
    parser.add_argument('--K', type=int, default=8, help='number of attention heads')
    parser.add_argument('--d', type=int, default=8, help='dims of each head attention outputs')
    parser.add_argument('--train_ratio', type=float, default=1, help='training set [default : 1]')
    parser.add_argument('--val_test_ratio', type=float, default=0.5, help='validation set [default : 0.5]')
    #parser.add_argument('--test_ratio', type=float, default=0.5, help='testing set [default : 1]')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--max_epoch', type=int, default=1, help='epoch to run')
    parser.add_argument('--patience', type=int, default=65, help='patience for early stop')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--decay_epoch', type=int, default=10, help='decay epoch')
    parser.add_argument('--train_x_file', default='./data/train_x_0.csv', help='train file with missing values')
    parser.add_argument('--train_y_file', default='./data/train_y_0.csv', help='train file with imputed values')
    parser.add_argument('--val_test_x_file', default='./data/val_test_x_0.csv', help='validation file with missing values')
    parser.add_argument('--val_test_y_file', default='./data/val_test_y_0.csv', help='validation file with imputed values')
    #parser.add_argument('--test_x_file', default='./masked/data/val_test.csv', help='test file with missing values')
    #parser.add_argument('--test_y_file', default='./masked/data/val_test.csv', help='test file with imputed values')
    parser.add_argument('--SE_file', default='./data/I-15_NB_SE.txt', help='spatial embedding file')
    parser.add_argument('--time_stamp', default='./data/timestamps_new.txt', help='time stamp file')
    parser.add_argument('--model_file', default='./data/model_I15NB_0_masked.pt', help='save the model to disk')
    parser.add_argument('--log_file', default='./data/log_I15NB_0_masked.txt', help='log file')
    parser.add_argument('--cuda_device', type=int, default=0, help='default CUDA device index if GPU is available')

    args = parser.parse_args()  # Parsing the command-line arguments

    # Determine if a GPU is available and set the device accordingly
    if torch.cuda.is_available():
        cuda_device = args.cuda_device
        torch.cuda.set_device(cuda_device)  # Setting the default CUDA device
        device = torch.device('cuda', cuda_device)  # Defining the device as CUDA
        device_str = f'Using CUDA device {cuda_device}: {torch.cuda.get_device_name(cuda_device)}'  # Device information string
    else:
        device = torch.device('cpu')  # Using CPU if CUDA is not available
        device_str = 'CUDA is not available. Using CPU.'  # Device information string

    log = open(args.log_file, 'w')  # Opening the log file in write mode
    log_string(log, str(args)[10: -1])  # Logging the parsed arguments except the program name
    log_string(log, device_str)  # Logging device information

    T = 24 * 60 // args.time_slot  # Number of time steps in one day

    # Loading data
    (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY, SE, mean, std) = load_data(args)

    # Logging data shapes and statistics
    log_string(log, f'trainX: {trainX.shape}\t\t trainY: {trainY.shape}')
    log_string(log, f'valX:   {valX.shape}\t\tvalY:   {valY.shape}')
    log_string(log, f'testX:   {testX.shape}\t\ttestY:   {testY.shape}')
    log_string(log, f'mean:   {mean:.4f}\t\tstd:   {std:.4f}')
    log_string(log, 'data loaded!')
    del trainX, trainTE, valX, valTE, testX, testTE, mean, std

    # Building the model
    log_string(log, 'compiling model...')
    model = GMAN(SE, args, bn_decay=0.1).to(device)  # Move model to the device
    loss_criterion = nn.MSELoss().to(device)  # Move loss criterion to the device

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)  # Defining the optimizer
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_epoch, gamma=0.9)  # Defining the learning rate scheduler
    parameters = count_parameters(model)  # Counting trainable parameters
    log_string(log, 'trainable parameters: {:,}'.format(parameters))  # Logging trainable parameters count

    start = time.time()  # Starting the timer
    loss_train, loss_val = train(model, args, log, loss_criterion, optimizer, scheduler)  # Training the model
    trainPred, valPred, testPred = test(args, log)  # Testing the model
    end = time.time()  # Ending the timer
    log_string(log, 'total time: %.1fmin' % ((end - start) / 60))  # Logging total execution time
    log.close()  # Closing the log file

    # Moving tensors to CPU and converting to NumPy arrays
    trainPred_ = trainPred.cpu().numpy().reshape(-1, trainY.shape[-1])
    trainY_ = trainY.cpu().numpy().reshape(-1, trainY.shape[-1])
    valPred_ = valPred.cpu().numpy().reshape(-1, valY.shape[-1])
    valY_ = valY.cpu().numpy().reshape(-1, valY.shape[-1])
    testPred_ = testPred.cpu().numpy().reshape(-1, testY.shape[-1])
    testY_ = testY.cpu().numpy().reshape(-1, testY.shape[-1])

    # Saving training, validation, and testing data to disk
    l = [trainPred_, trainY_, valPred_, valY_, testPred_, testY_]
    name = ['trainPred', 'trainY', 'valPred', 'valY', 'testPred', 'testY']
    for i, data in enumerate(l):
        np.savetxt('./masked/figure/' + name[i] + '.txt', data, fmt='%s')

    print('Done!')

if __name__ == '__main__':
    main()
