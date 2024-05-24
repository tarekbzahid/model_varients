import torch
import time
import math
import numpy as np
from utils.utils_ import log_string, metric
from utils.utils_ import load_data

# Assuming device is your CUDA device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test(args, log):
    (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE,
     testY, SE, mean, std) = load_data(args)
    
    
    if torch.cuda.is_available():
        trainX = trainX.to(device)
        trainTE = trainTE.to(device)
        trainY = trainY.to(device)
        valX = valX.to(device)
        valTE = valTE.to(device)
        valY = valY.to(device)
        testX = testX.to(device)
        testTE = testTE.to(device)
        testY = testY.to(device)
        SE= SE.to(device)
        mean = mean.to(device)
        std = std.to(device)


    num_train, _, num_vertex = trainX.shape
    num_val = valX.shape[0]
    num_test = testX.shape[0]
    train_num_batch = math.ceil(num_train / args.batch_size)
    val_num_batch = math.ceil(num_val / args.batch_size)
    test_num_batch = math.ceil(num_test / args.batch_size)

    model = torch.load(args.model_file)

    # test model
    log_string(log, '**** testing model ****')
    log_string(log, 'loading model from %s' % args.model_file)
    model = torch.load(args.model_file)
    log_string(log, 'model restored!')
    log_string(log, 'evaluating...')

    with torch.no_grad():

        #trainPred = []
        trainPred = torch.empty(0, device=device)  
        for batch_idx in range(train_num_batch):
            start_idx = batch_idx * args.batch_size
            end_idx = min(num_train, (batch_idx + 1) * args.batch_size)
            X = trainX[start_idx: end_idx]
            TE = trainTE[start_idx: end_idx]
            pred_batch = model(X, TE)
            #trainPred.append(pred_batch.detach().clone())
            trainPred = torch.cat((trainPred, pred_batch.detach().clone()), dim=0)
            del X, TE, pred_batch
        trainPred = trainPred * std + mean


        #valPred = []
        valPred = torch.empty(0, device=device)
        for batch_idx in range(val_num_batch):
            start_idx = batch_idx * args.batch_size
            end_idx = min(num_val, (batch_idx + 1) * args.batch_size)
            X = valX[start_idx: end_idx]
            TE = valTE[start_idx: end_idx]
            pred_batch = model(X, TE)
            #valPred.append(pred_batch.detach().clone())
            valPred = torch.cat((valPred, pred_batch.detach().clone()), dim=0)
            del X, TE, pred_batch
        #valPred = torch.from_numpy(np.concatenate(valPred, axis=0))
        valPred = valPred * std + mean

        #testPred = []
        testPred = torch.empty(0, device=device)
        start_test = time.time()
        for batch_idx in range(test_num_batch):
            start_idx = batch_idx * args.batch_size
            end_idx = min(num_test, (batch_idx + 1) * args.batch_size)
            X = testX[start_idx: end_idx]
            TE = testTE[start_idx: end_idx]
            pred_batch = model(X, TE)
            #testPred.append(pred_batch.detach().clone())
            testPred = torch.cat((testPred, pred_batch.detach().clone()), dim=0)
            del X, TE, pred_batch
        #testPred = torch.from_numpy(np.concatenate(testPred, axis=0))
        testPred = testPred* std + mean
    end_test = time.time()
    
    train_mae, train_rmse, train_mape = metric(trainPred, trainY)
    val_mae, val_rmse, val_mape = metric(valPred, valY)
    test_mae, test_rmse, test_mape = metric(testPred, testY)
    log_string(log, 'testing time: %.1fs' % (end_test - start_test))
    log_string(log, '                MAE\t\tRMSE\t\tMAPE')
    log_string(log, 'train            %.2f\t\t%.2f\t\t%.2f%%' %
               (train_mae, train_rmse, train_mape * 100))
    log_string(log, 'val              %.2f\t\t%.2f\t\t%.2f%%' %
               (val_mae, val_rmse, val_mape * 100))
    log_string(log, 'test             %.2f\t\t%.2f\t\t%.2f%%' %
               (test_mae, test_rmse, test_mape * 100))
    log_string(log, 'performance in each prediction step')
    
    MAE, RMSE, MAPE = [], [], []

    for step in range(args.num_pred):
        mae, rmse, mape = metric(testPred[:, step], testY[:, step])
        MAE.append(mae)
        RMSE.append(rmse)
        MAPE.append(mape)
        log_string(log, 'step: %02d         %.2f\t\t%.2f\t\t%.2f%%' %
               (step + 1, mae, rmse, mape * 100))

    # Convert lists to tensors
    MAE_tensor = torch.tensor(MAE, device=device)
    RMSE_tensor = torch.tensor(RMSE, device=device)
    MAPE_tensor = torch.tensor(MAPE, device=device)

    # Calculate the average metrics
    average_mae = torch.mean(MAE_tensor).item()
    average_rmse = torch.mean(RMSE_tensor).item()
    average_mape = torch.mean(MAPE_tensor).item()

    # Log average metrics
    log_string(log, f'Average MAE: {average_mae:.2f}')
    log_string(log, f'Average RMSE: {average_rmse:.2f}')
    log_string(log, f'Average MAPE: {average_mape:.2f}%')

    return trainPred, valPred, testPred
