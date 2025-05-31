import torch 
import torch.nn as nn
import pandas as pd 
import os 
from loguru import logger 


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=False):
        """
        Early stopping to stop the training when the loss does not improve after
        certain epochs.
        
        Args:
            patience (int): How many epochs to wait before stopping when loss is
                           not improving
            min_delta (float): Minimum change in the monitored quantity to
                             qualify as an improvement
            verbose (bool): If True, prints a message for each validation loss improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        
    def __call__(self, val_loss, model, optimizer, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, optimizer, epoch)
        elif val_loss > self.best_loss + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, optimizer, epoch)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model, optimizer, epoch):
        """Saves model when validation loss decreases."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss




def create_dataset(dataset, lookback,split_index):
    """Transform a time series into a prediction dataset

    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """


    X_train, y_train = [], []
    X_test, y_test = [], []
    for i in range(lookback, len(dataset)):
        if i < split_index:
            feature = dataset[i-lookback:i]
            target = dataset[i:i+1][:,0:1]
            X_train.append(feature)
            y_train.append(target)
        else:
            feature = dataset[i-lookback:i]
            target = dataset[i:i+1][:,0:1]
            X_test.append(feature)
            y_test.append(target)
    # X = np.array(X)
    # y = np.array(y)
    return torch.tensor(X_train), torch.tensor(y_train),torch.tensor(X_test), torch.tensor(y_test)



def train(name,ALL_MODEL,timeseries,first_index,end_index,steps,LOOKBACK,BATCHSIZE):
    import numpy as np
    import torch.optim as optim
    import torch.utils.data as data
    import time 
    begin_time = time.time()
    df_result = pd.DataFrame()

    for split_index in range(first_index,end_index,steps): #, len(timeseries)
        if os.path.exists(f'./data/{name}_result_{LOOKBACK}_{BATCHSIZE}_{first_index}_{end_index}.csv'):
            df_result_exists = pd.read_csv(f'./data/{name}_result_{LOOKBACK}_{BATCHSIZE}_{first_index}_{end_index}.csv')
            df_result_exists = df_result_exists[df_result_exists['true']!='true']
            df_result_exists['split_index'] = df_result_exists['split_index'].astype(int)
            if split_index in df_result_exists['split_index'].values:
                logger.info(f'{name} {LOOKBACK} {BATCHSIZE} {first_index} {end_index} {split_index} 已经存在')
            # if len(df_result_exists) > 0:
                continue
        else:
            logger.info(f'{name} {LOOKBACK} {BATCHSIZE} {first_index} {end_index} {split_index} 不存在')

        
        train_size = split_index # 
        # test_size = len(timeseries) - train_size
        # train, test = timeseries[:train_size], timeseries[train_size:]
        lookback = LOOKBACK 
        X_train_all, y_train_all,X_test, y_test = create_dataset(timeseries, lookback=lookback,split_index=split_index)
        # X_test, y_test = create_dataset(test, lookback=lookback)
        print(X_train_all.shape, y_train_all.shape)
        print(X_test.shape, y_test.shape)


        indices = np.arange(len(X_train_all))   
        np.random.shuffle(indices)
        train_val_size = int(len(X_train_all) * 0.8)
        train_indices = indices[:train_val_size]
        val_indices = indices[train_val_size:]
        X_train = X_train_all[train_indices]
        y_train = y_train_all[train_indices]
        X_val = X_train_all[val_indices]
        y_val = y_train_all[val_indices]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        X_train = X_train.to(device)
        y_train = y_train.to(device)
        X_val = X_val.to(device)
        y_val = y_val.to(device)
        X_test = X_test.to(device)
        y_test = y_test.to(device)

        # model = TrendAwareLSTM().to(device)
        if name == "FCNN":
            X_train = X_train.squeeze(-1)
            y_train = y_train.squeeze(-1)
            X_val = X_val.squeeze(-1)
            y_val = y_val.squeeze(-1)
            X_test = X_test.squeeze(-1)
            y_test = y_test.squeeze(-1)
            model = ALL_MODEL[name](input_size=100, hidden_size=100, output_size=1).to(device)
        else:
            model = ALL_MODEL[name]().to(device)
        # model = model.to(device)


        optimizer = optim.Adam(model.parameters())
        loss_fn = nn.MSELoss()
        loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=BATCHSIZE)
        early_stopping = EarlyStopping(patience=10, min_delta=0.001)
        n_epochs = 1000
        for epoch in range(n_epochs):
            model.train()
            for X_batch, y_batch in loader:
                # print(X_batch.shape)
                # X_batch = X_batch.to(device)
                # y_batch = y_batch.to(device)
                y_pred = model(X_batch)
                y_pred_val = model(X_val)
                # print(y_pred.shape,y_batch.shape)
                loss_train = loss_fn(y_pred, y_batch)
                
                optimizer.zero_grad()
                loss_train.backward()
                optimizer.step()
            # Validation
            loss_val = loss_fn(y_pred_val, y_val)
            early_stopping(loss_val, model, optimizer, epoch)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        model.eval()
        with torch.no_grad():
            pass 
            # X_train = X_train.to(device)
            # y_train = y_train.to(device)
            # y_pred_train = model(X_train)
            # X_test = X_test.to(device)
            # y_test = y_test.to(device)
            # train_rmse = torch.sqrt(loss_fn(y_pred_train, y_train))
            # y_pred_test = model(X_test)
            # test_rmse = torch.sqrt(loss_fn(y_pred_test, y_test))
        # print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))
        end_time = time.time()
        print(f"训练时间: {end_time - begin_time} 秒")
        if name == "FCNN":
            predict_value = model(X_test)
            predict_value = predict_value.detach().cpu().numpy().flatten()
        else:
            predict_value = model(X_test[0,:,:].unsqueeze(0))
            predict_value = predict_value.detach().cpu().numpy().flatten()


        df_result_tmp = pd.DataFrame([[split_index, predict_value[0],y_test[0].flatten().cpu().numpy()[0]]],columns=['split_index','predict','true'])
        df_result_tmp
        # df_result = pd.concat([df_result,df_result_tmp],ignore_index=True)
        # name = model.__class__.__name__
        df_result_tmp.to_csv(f'./data/{name}_result_{LOOKBACK}_{BATCHSIZE}_{first_index}_{end_index}.csv',index=False,mode='a')
        
        del model
        del optimizer
        del loader
        del early_stopping
        del X_train
        del y_train
        del X_train_all
        del y_train_all
        del X_val
        del y_val
        del X_test
        del y_test
        # del y_pred_train
        # del y_pred_test
        del y_pred_val
        del y_pred
        del y_batch
        del X_batch
        torch.cuda.empty_cache()




def process_data():
    df = pd.read_csv('./data/df.csv',sep=',')
    df  = df.sort_values(by='time',ascending=True).reset_index(drop=True)

    # BATCHSIZE = 256
    # LOOKBACK = 100


    date = ["2024-02-19",
    "2024-03-15",
    "2024-04-19",
    "2024-05-17",
    "2024-06-21",
    "2024-07-19",
    "2024-08-16",
    "2024-09-20",
    "2024-10-18",
    "2024-11-15",
    "2024-12-20",
    "2025-01-17"]
    df.head()

    df['time'] = pd.to_datetime(df['time'])
    df['date'] = df['time'].dt.date
    df['date'] = df['date'].astype(str)
    df['hour'] = df['time'].dt.hour
    df['minute'] = df['time'].dt.minute 
    # for i in range(1,5):
    #     df[f'spread_shift_{i}'] = df.groupby(['hour','minute'])['spread'].shift(i)

    df['Expiration_Date'] = df['date'].apply(lambda x: 1 if x in date else 0)
    # df.fillna(0,inplace=True)


    df.dropna(inplace=True)

    df.head()
    return df 

        
