#from cupshelpers import Printer
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


# log string
def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)


# metric
def metric(pred, label):
    mask = torch.ne(label, 0)
    mask = mask.type(torch.float32)
    mask /= torch.mean(mask)
    mae = torch.abs(torch.sub(pred, label)).type(torch.float32)
    rmse = mae ** 2
    mape = mae / label
    mae = torch.mean(mae)
    rmse = rmse * mask
    rmse = torch.sqrt(torch.mean(rmse))
    mape = mape * mask
    mape = torch.mean(mape)
    return mae, rmse, mape


def seq2instance(data, num_his, num_pred):
    num_step, dims = data.shape
    num_sample = num_step - num_his - num_pred + 1
    x = torch.zeros(num_sample, num_his, dims)
    y = torch.zeros(num_sample, num_pred, dims)
    for i in range(num_sample):
        x[i] = data[i: i + num_his]
        y[i] = data[i + num_his: i + num_his + num_pred]
    return x, y


def load_data(args):
    # Reading CSV train-val-test data files
    train = pd.read_csv(args.train_file)
    train=torch.from_numpy(train.values)
    val_test= pd.read_csv(args.val_test_file)
    val_test=torch.from_numpy(val_test.values)

    # train/val/test
    num_step_train = train.shape[0]
    num_step_val_test=val_test.shape[0]

    train_steps = round(args.train_ratio * num_step_train)

    test_steps = round(args.test_ratio * num_step_val_test)
    val_steps = num_step_val_test - test_steps

    train = train[: train_steps]
    val = val_test[: val_steps]
    test = val_test[-test_steps:]
    
    # X, Y
    trainX, trainY = seq2instance(train, args.num_his, args.num_pred)
    valX, valY = seq2instance(val, args.num_his, args.num_pred)
    testX, testY = seq2instance(test, args.num_his, args.num_pred)

    # normalization
    mean, std = torch.mean(trainX), torch.std(trainX)
    trainX = (trainX - mean) / std
    valX = (valX - mean) / std
    testX = (testX - mean) / std

    # spatial embedding
    with open(args.SE_file, mode='r') as f:
        lines = f.readlines()
        temp = lines[0].split(' ')
        #print("temp",temp)
        num_vertex, dims = int(temp[0]), int(temp[1])
        #print("num vertex", num_vertex)
        #print("dims", dims)
        SE = torch.zeros((num_vertex, dims), dtype=torch.float32)
        #print("SE", SE)

        for line in lines[1:]:
            temp = line.split(' ')
            index = int(temp[0])
            SE[index-1] = torch.tensor([float(ch) for ch in temp[1:]])
            

    # temporal embedding
    # the following code is being replaced temporariliy for testing with a test code
    # remove 1 # to get the original code:

    #time = pd.DatetimeIndex(df.index)
    #dayofweek = torch.reshape(torch.tensor(time.weekday), (-1, 1))
    #timeofday = (time.hour * 3600 + time.minute * 60 + time.second) // (300)

    ##            // time.freq.delta.total_seconds()
    #timeofday = torch.reshape(torch.tensor(timeofday), (-1, 1))
    #time = torch.cat((dayofweek, timeofday), -1)
    # Read the CSV file into a DataFrame
    
    # --------------------------------------------------------
    # Read the datetime values from the text file
    with open(args.time_stamp, 'r') as file:
        datetime_strings = file.read().splitlines()

    # Convert the datetime strings to pandas DatetimeIndex
    time = pd.DatetimeIndex(datetime_strings)

    # Extract day of the week
    dayofweek = torch.reshape(torch.tensor(time.weekday), (-1, 1))

    # Calculate time of day in minutes
    timeofday = torch.tensor((time.hour * 60 + time.minute).values).reshape(-1, 1)

    # Concatenate dayofweek and timeofday tensors
    time = torch.cat((dayofweek, timeofday), dim=-1)
    # --------------------------------------------------------

    # train/val/test
    train = time[: train_steps]
    val = time[: val_steps]
    test = time[-test_steps:]
    # shape = (num_sample, num_his + num_pred, 2)
    trainTE = seq2instance(train, args.num_his, args.num_pred)
    trainTE = torch.cat(trainTE, 1).type(torch.int32)
    valTE = seq2instance(val, args.num_his, args.num_pred)
    valTE = torch.cat(valTE, 1).type(torch.int32)
    testTE = seq2instance(test, args.num_his, args.num_pred)
    testTE = torch.cat(testTE, 1).type(torch.int32)

    return (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY,
            SE, mean, std)


# dataset creation
class dataset(Dataset):
    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y
        self.len = data_x.shape[0]

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]

    def __len__(self):
        return self.len


# statistic model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# The following function can be replaced by 'loss = torch.nn.L1Loss()  loss_out = loss(pred, target)
def mae_loss(pred, label):
    mask = torch.ne(label, 0)
    mask = mask.type(torch.float32)
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.tensor(0.0), mask)
    loss = torch.abs(torch.sub(pred, label))
    loss *= mask
    loss = torch.where(torch.isnan(loss), torch.tensor(0.0), loss)
    loss = torch.mean(loss)
    return loss


# plot train_val_loss
#def plot_train_val_loss(train_total_loss, val_total_loss, file_path):
    #plt.figure(figsize=(10, 5))
    #plt.plot(range(1, len(train_total_loss) + 1), train_total_loss, c='b', marker='s', label='Train')
    #plt.plot(range(1, len(val_total_loss) + 1), val_total_loss, c='r', marker='o', label='Validation')
    #plt.legend(loc='best')
    #plt.title('Train loss vs Validation loss')
    #plt.savefig(file_path)

def plot_train_val_loss(train_total_loss, val_total_loss, file_path):
    # Move tensors to CPU and convert to NumPy arrays if they are CUDA tensors
    if isinstance(train_total_loss, torch.Tensor):
        train_total_loss = train_total_loss.cpu().detach().numpy()
    if isinstance(val_total_loss, torch.Tensor):
        val_total_loss = val_total_loss.cpu().detach().numpy()
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_total_loss) + 1), train_total_loss, c='b', marker='s', label='Train')
    plt.plot(range(1, len(val_total_loss) + 1), val_total_loss, c='r', marker='o', label='Validation')
    plt.legend(loc='best')
    plt.title('Train loss vs Validation loss')
    plt.savefig(file_path)



# plot test results
def save_test_result(trainPred, trainY, valPred, valY, testPred, testY):
    with open('./figure/test_results.txt', 'w+') as f:
        for l in (trainPred, trainY, valPred, valY, testPred, testY):
            f.write(list(l))


def check_gpu_usage(model):
    print("Checking GPU usage:")
    for name, param in model.named_parameters():
        if param.requires_grad and param.is_cuda:
            print(f"{name} is on GPU")
        elif param.requires_grad and not param.is_cuda:
            print(f"{name} is NOT on GPU")
