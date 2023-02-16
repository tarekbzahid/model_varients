import torch
import time
import math
import numpy as np
from utils.utils_ import log_string, metric
from utils.utils_ import load_data

def test(args, log):
    (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE,
     testY, SE, mean, std) = load_data(args)
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

        testPred = []
        start_test = time.time()
        #for batch_idx in range(test_num_batch):
            #start_idx = batch_idx * args.batch_size
            #end_idx = min(num_test, (batch_idx + 1) * args.batch_size)
            #X = testX[start_idx: end_idx]
            #TE = testTE[start_idx: end_idx]
        X = testX
        TE = testTE
        pred_batch = model(X, TE)
        testPred.append(pred_batch.detach().clone())
        del X, TE, pred_batch
        testPred = torch.from_numpy(np.concatenate(testPred, axis=0))
        testPred = testPred* std + mean
    end_test = time.time()
    #train_mae, train_rmse, train_mape = metric(trainPred, trainY)
    #val_mae, val_rmse, val_mape = metric(valPred, valY)
    test_mae, test_rmse, test_mape = metric(testPred, testY)
    log_string(log, 'testing time: %.1fs' % (end_test - start_test))
    log_string(log, '                MAE\t\tRMSE\t\t')

    #log_string(log, 'test            %.2f\t\t%.2f\t\t' %
               #(test_mae, test_rmse))
    log_string(log, 'Final Test Performance')
    MAE, RMSE, MAPE = [], [], []
    for step in range(args.num_pred):
        mae, rmse, mape = metric(testPred[:, step], testY[:, step])
        MAE.append(mae)
        RMSE.append(rmse)
        MAPE.append(mape)
        log_string(log, 'step: %02d         %.2f\t\t%.2f\t\t' %
                   (step + 1, mae, rmse))
    average_mae = np.mean(MAE)
    average_rmse = np.mean(RMSE)
    average_mape = np.mean(MAPE)
    log_string(
        log, 'average:         %.2f\t\t%.2f\t\t' %
             (average_mae, average_rmse))
    return testPred
