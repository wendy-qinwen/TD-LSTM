import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yaml
import os
import torch.utils.data as data
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def load_config(config_path="config.yaml"):
    """
    加载YAML配置文件
    
    Args:
        config_path: YAML配置文件路径
    
    Returns:
        config: 配置字典
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件 {config_path} 不存在")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config

def calculate_spread_volatility(spread_data, window=20, method='log_returns'):
    """
    计算价差波动率
    
    Args:
        spread_data: 价差数据，numpy数组或pandas Series
        window: 计算窗口大小
        method: 计算方法，可选 'log_returns', 'absolute', 'relative'
    
    Returns:
        价差波动率序列
    """
    if isinstance(spread_data, pd.Series):
        spread_data = spread_data.values
    
    if method == 'log_returns':
        # 对数收益率波动率
        log_returns = np.log(spread_data[1:] / spread_data[:-1])
        volatility = pd.Series(log_returns).rolling(window=window).std() * np.sqrt(252)  # 年化
        return np.insert(volatility.values, 0, np.nan)
    
    elif method == 'absolute':
        # 绝对价差波动率
        spread_changes = np.diff(spread_data)
        volatility = pd.Series(spread_changes).rolling(window=window).std() * np.sqrt(252)  # 年化
        return np.insert(volatility.values, 0, np.nan)
    
    elif method == 'relative':
        # 相对价差波动率
        relative_changes = np.diff(spread_data) / spread_data[:-1]
        volatility = pd.Series(relative_changes).rolling(window=window).std() * np.sqrt(252)  # 年化
        return np.insert(volatility.values, 0, np.nan)
    
    else:
        raise ValueError(f"不支持的价差波动率计算方法: {method}")

def calculate_volatility(data, window=20, method='historical'):
    """
    计算波动率
    
    Args:
        data: 价格数据，numpy数组或pandas Series
        window: 计算窗口大小
        method: 计算方法，可选 'historical', 'log_returns', 'parkinson', 'garman_klass'
    
    Returns:
        波动率序列
    """
    if isinstance(data, pd.Series):
        data = data.values
    
    if method == 'historical':
        # 历史波动率
        returns = np.diff(data) / data[:-1]
        volatility = pd.Series(returns).rolling(window=window).std() * np.sqrt(252)  # 年化
        return volatility.values
    
    elif method == 'log_returns':
        # 对数收益率波动率
        log_returns = np.log(data[1:] / data[:-1])
        volatility = pd.Series(log_returns).rolling(window=window).std() * np.sqrt(252)  # 年化
        return np.insert(volatility.values, 0, np.nan)
    
    elif method == 'parkinson':
        # Parkinson波动率（使用最高价和最低价）
        if isinstance(data, pd.DataFrame):
            high = data['high'].values
            low = data['low'].values
        else:
            high = data
            low = data
        
        log_hl = np.log(high[1:] / low[1:])
        volatility = pd.Series(log_hl).rolling(window=window).apply(
            lambda x: np.sqrt(1/(4*np.log(2))*np.mean(x**2)) * np.sqrt(252)  # 年化
        )
        return np.insert(volatility.values, 0, np.nan)
    
    elif method == 'garman_klass':
        # Garman-Klass波动率（使用OHLC数据）
        if isinstance(data, pd.DataFrame):
            open_price = data['open'].values
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
        else:
            open_price = data
            high = data
            low = data
            close = data
        
        log_hl = np.log(high[1:] / low[1:])
        log_co = np.log(close[1:] / open_price[1:])
        
        volatility = pd.Series(
            np.sqrt(0.5 * log_hl**2 - (2*np.log(2)-1) * log_co**2)
        ).rolling(window=window).mean() * np.sqrt(252)  # 年化
        
        return np.insert(volatility.values, 0, np.nan)
    
    else:
        raise ValueError(f"不支持的波动率计算方法: {method}")

def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset

    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback]
        target = dataset[i+lookback:i+lookback+1]
        X.append(feature)
        y.append(target)
    
    # 将列表转换为numpy数组，然后再转换为张量，并指定数据类型为float32
    X = np.array(X)
    y = np.array(y)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class AirModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        
        # 如果没有提供配置，加载默认配置
        if config is None:
            config = load_config()
        
        # 从配置中获取模型参数
        model_config = config['model']
        lstm1_config = model_config['lstm1']
        lstm2_config = model_config['lstm2']
        linear_config = model_config['linear']
        
        # 创建LSTM层和线性层
        self.lstm = nn.LSTM(
            input_size=lstm1_config['input_size'], 
            hidden_size=lstm1_config['hidden_size'], 
            num_layers=lstm1_config['num_layers'], 
            batch_first=lstm1_config['batch_first'],
            dropout=lstm1_config['dropout']
        )
        
        self.lstm2 = nn.LSTM(
            input_size=lstm2_config['input_size'], 
            hidden_size=lstm2_config['hidden_size'], 
            num_layers=lstm2_config['num_layers'], 
            batch_first=lstm2_config['batch_first'],
            dropout=lstm2_config['dropout']
        )
        
        self.linear = nn.Linear(
            linear_config['input_size'],
            linear_config['output_size']
        )
    
    def forward(self, x):
        # 第一个LSTM层
        x, _ = self.lstm(x)  # x shape: [batch_size, seq_len, hidden_size]
        # 第二个LSTM层
        x, _ = self.lstm2(x)  # x shape: [batch_size, seq_len, hidden_size]
        # 使用整个序列的输出
        x = self.linear(x)  # x shape: [batch_size, seq_len, 1]
        # 增加一个维度
        x = x.unsqueeze(-1)  # x shape: [batch_size, seq_len, 1, 1]
        return x

# 创建模型实例
def create_model(config_path="config.yaml"):
    config = load_config(config_path)
    return AirModel(config)

model = create_model()

# 定义损失函数和优化器
def create_criterion_optimizer(model, config_path="config.yaml"):
    config = load_config(config_path)
    training_config = config['training']
    
    # 创建损失函数
    loss_function = training_config['loss_function'].lower()
    if loss_function == 'mse':
        criterion = nn.MSELoss()
    elif loss_function == 'mae':
        criterion = nn.L1Loss()
    elif loss_function == 'huber':
        criterion = nn.SmoothL1Loss()
    else:
        raise ValueError(f"不支持的损失函数: {loss_function}")
    
    # 创建优化器
    optimizer_name = training_config['optimizer'].lower()
    lr = training_config['learning_rate']
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError(f"不支持的优化器: {optimizer_name}")
    
    return criterion, optimizer

criterion, optimizer = create_criterion_optimizer(model)

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

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, config):
    """
    训练模型
    
    Args:
        model: LSTM模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        num_epochs: 训练轮数
        device: 计算设备
        config: 配置字典
    """
    model.train()
    early_stopping = EarlyStopping(
        patience=config['training']['early_stopping_patience'],
        min_delta=config['training']['early_stopping_min_delta'],
        verbose=True
    )
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            # 清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 确保数据类型一致
            batch_x = batch_x.float()
            batch_y = batch_y.float()
            
            # 重塑输入张量以匹配模型期望的形状
            # 假设模型期望的输入形状是 [batch_size, input_features]
            # 而batch_x的形状是 [batch_size, seq_len, features]
            batch_size, seq_len, features = batch_x.shape
            batch_x = batch_x.reshape(batch_size, seq_len * features)
            
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 释放不需要的张量
            del outputs
            del loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
        
        # 验证损失
        with torch.no_grad():
            val_loss = 0
            for val_x, val_y in val_loader:
                # 确保数据类型一致
                val_x = val_x.float()
                val_y = val_y.float()
                
                # 重塑验证集输入张量
                val_batch_size, val_seq_len, val_features = val_x.shape
                val_x = val_x.reshape(val_batch_size, val_seq_len * val_features)
                
                val_x = val_x.to(device)
                val_y = val_y.to(device)
                val_outputs = model(val_x)
                val_loss += criterion(val_outputs, val_y).item()
                del val_outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            val_loss = val_loss / len(val_loader)
        
        early_stopping(val_loss, model, optimizer, epoch)
        
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    # 加载最佳模型
    model.load_state_dict(torch.load('checkpoint.pt'))

# 预测函数
def predict(model, x):
    model.eval()
    with torch.no_grad():
        return model(x)

def concat_tensors(tensor1, tensor2, dim=-1):
    """
    拼接两个张量
    
    Args:
        tensor1: 第一个张量，形状为 [256, 5, 1]
        tensor2: 第二个张量，形状为 [256, 5, 3]
        dim: 拼接维度，默认为-1（最后一个维度）
    
    Returns:
        拼接后的张量，形状为 [256, 5, 4]
    """
    # 确保输入是张量
    if not isinstance(tensor1, torch.Tensor):
        tensor1 = torch.tensor(tensor1)
    if not isinstance(tensor2, torch.Tensor):
        tensor2 = torch.tensor(tensor2)
    
    # 检查形状
    assert tensor1.shape[:2] == tensor2.shape[:2], "前两个维度必须相同"
    
    # 在最后一个维度上拼接
    concatenated = torch.cat([tensor1, tensor2], dim=dim)
    return concatenated

def count_parameters(model):
    """
    计算模型的总参数量
    
    Args:
        model: PyTorch模型
    
    Returns:
        total_params: 总参数量
        trainable_params: 可训练参数量
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"模型结构:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape} = {param.numel()} 参数")
    
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    return total_params, trainable_params

# 使用示例
def example_usage(config_path="config.yaml"):
    config = load_config(config_path)
    
    # 创建示例张量
    tensor1 = torch.randn(256, 5, 1)  # 形状: [256, 5, 1]
    tensor2 = torch.randn(256, 5, 3)  # 形状: [256, 5, 3]
    
    # 拼接张量
    result = concat_tensors(tensor1, tensor2)
    print(f"拼接后的张量形状: {result.shape}")  # 应该输出: torch.Size([256, 5, 4])
    
    # 创建模型
    model = create_model(config_path)
    
    # 计算模型参数量
    total_params, trainable_params = count_parameters(model)
    
    return result 

# 使用配置创建数据集和加载器
def prepare_data(timeseries, config_path="config.yaml"):
    """
    准备训练、验证和测试数据
    
    Args:
        timeseries: 时间序列数据
        config_path: 配置文件路径
    
    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据
        X_test, y_test: 测试数据
    """
    config = load_config(config_path)
    data_config = config['data']
    training_config = config['training']
    
    # 数据参数
    lookback = data_config['lookback']
    train_test_split = data_config['train_test_split']
    shuffle = data_config['shuffle']
    batch_size = training_config['batch_size']
    
    # 确保输入数据是float32类型
    timeseries = timeseries.astype(np.float32)
    
    # 训练集测试集分割
    train_size = int(len(timeseries) * train_test_split)
    train_data, test_data = timeseries[:train_size], timeseries[train_size:]
    
    # 随机选择训练集和验证集
    indices = np.arange(len(train_data))
    np.random.shuffle(indices)  # 随机打乱索引
    
    train_val_split = 0.8  # 80% 训练，20% 验证
    train_val_size = int(len(train_data) * train_val_split)
    
    # 使用随机索引分割数据
    train_indices = indices[:train_val_size]
    val_indices = indices[train_val_size:]
    
    train_data = train_data[train_indices]
    val_data = train_data[val_indices]
    
    # 创建数据集
    X_train, y_train = create_dataset(train_data, lookback=lookback)
    X_val, y_val = create_dataset(val_data, lookback=lookback)
    X_test, y_test = create_dataset(test_data, lookback=lookback)
    
    print(f"训练集形状: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"验证集形状: X_val={X_val.shape}, y_val={y_val.shape}")
    print(f"测试集形状: X_test={X_test.shape}, y_test={y_test.shape}")
    
    # 创建数据加载器
    train_dataset = data.TensorDataset(X_train, y_train)
    val_dataset = data.TensorDataset(X_val, y_val)
    
    train_loader = data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle
    )
    
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False  # 验证集不需要打乱
    )
    
    return train_loader, val_loader, X_train, y_train, X_val, y_val, X_test, y_test

def main(config_path="config.yaml"):
    """
    主函数，演示完整的训练和评估流程
    
    Args:
        config_path: 配置文件路径
    """
    try:
        # 加载配置
        config = load_config(config_path)
        
        # 读取数据
        print("正在读取数据...")
        df = pd.read_csv('./data/df.csv', sep=',')
        timeseries = df[["spread"]].values.astype('float32')  # 确保数据类型为float32
        
        # 准备数据
        print("正在准备数据...")
        train_loader, val_loader, X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(timeseries, config_path)
        
        # 创建模型
        print("正在创建模型...")
        model = create_model(config_path)
        
        # 确保模型参数是float32类型
        model = model.float()
        
        # 计算模型参数
        print("模型参数统计:")
        count_parameters(model)
        
        # 训练模型
        print("正在训练模型...")
        train_model(model, train_loader, val_loader, criterion, optimizer, config['training']['num_epochs'], torch.device("cuda" if torch.cuda.is_available() else "cpu"), config)
        
        # 评估模型
        print("正在评估模型...")
        model.eval()
        
        # 移动数据到设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # 重塑数据以匹配模型期望的形状
        batch_size, seq_len, features = X_train.shape
        X_train_reshaped = X_train.reshape(batch_size, seq_len * features)
        
        batch_size, seq_len, features = X_val.shape
        X_val_reshaped = X_val.reshape(batch_size, seq_len * features)
        
        batch_size, seq_len, features = X_test.shape
        X_test_reshaped = X_test.reshape(batch_size, seq_len * features)
        
        X_train_reshaped = X_train_reshaped.to(device)
        y_train = y_train.to(device)
        X_val_reshaped = X_val_reshaped.to(device)
        y_val = y_val.to(device)
        X_test_reshaped = X_test_reshaped.to(device)
        y_test = y_test.to(device)
        
        # 计算损失
        criterion, _ = create_criterion_optimizer(model, config_path)
        
        with torch.no_grad():
            # 训练集预测
            y_train_pred = model(X_train_reshaped)
            y_train_pred = y_train_pred.squeeze(-1)
            train_loss = criterion(y_train_pred, y_train).item()
            del y_train_pred
            
            # 验证集预测
            y_val_pred = model(X_val_reshaped)
            y_val_pred = y_val_pred.squeeze(-1)
            val_loss = criterion(y_val_pred, y_val).item()
            del y_val_pred
            
            # 测试集预测
            y_test_pred = model(X_test_reshaped)
            y_test_pred = y_test_pred.squeeze(-1)
            test_loss = criterion(y_test_pred, y_test).item()
            del y_test_pred
        
        print(f"训练集损失: {train_loss:.4f}")
        print(f"验证集损失: {val_loss:.4f}")
        print(f"测试集损失: {test_loss:.4f}")
        
        # 可视化结果
        print("正在可视化结果...")
        visualize_results(timeseries, X_train, X_val, X_test, y_train_pred, y_val_pred, y_test_pred, config)
        
        return model
        
    finally:
        # 清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 删除不需要的变量
        del model
        del train_loader
        del val_loader
        del X_train
        del y_train
        del X_val
        del y_val
        del X_test
        del y_test
        del criterion
        del optimizer

def visualize_results(timeseries, X_train, X_val, X_test, y_train_pred, y_val_pred, y_test_pred, config):
    """
    可视化预测结果
    
    Args:
        timeseries: 原始时间序列
        X_train, X_val, X_test: 训练集、验证集和测试集特征
        y_train_pred, y_val_pred, y_test_pred: 训练集、验证集和测试集预测结果
        config: 配置
    """
    # 获取配置
    data_config = config['data']
    lookback = data_config['lookback']
    train_test_split = data_config['train_test_split']
    
    # 移动数据到CPU
    y_train_pred = y_train_pred.cpu().numpy()
    y_val_pred = y_val_pred.cpu().numpy()
    y_test_pred = y_test_pred.cpu().numpy()
    
    # 计算训练、验证和测试的索引
    train_size = int(len(timeseries) * train_test_split)
    train_val_size = int(train_size * 0.8)  # 80% 训练，20% 验证
    
    # 准备绘图数据
    train_plot = np.ones_like(timeseries) * np.nan
    val_plot = np.ones_like(timeseries) * np.nan
    test_plot = np.ones_like(timeseries) * np.nan
    
    # 填充预测值
    train_plot[lookback:train_val_size] = y_train_pred[:, -1, :]
    val_plot[train_val_size+lookback:train_size] = y_val_pred[:, -1, :]
    test_plot[train_size+lookback:len(timeseries)] = y_test_pred[:, -1, :]
    
    # 创建时间索引
    time_index = np.arange(len(timeseries))
    
    # Matplotlib绘图
    plt.figure(figsize=(12, 6))
    plt.plot(time_index, timeseries, 'b', label='原始数据')
    plt.plot(time_index, train_plot, 'r', label='训练集预测')
    plt.plot(time_index, val_plot, 'y', label='验证集预测')
    plt.plot(time_index, test_plot, 'g', label='测试集预测')
    plt.title('LSTM 预测结果')
    plt.xlabel('时间步')
    plt.ylabel('价差')
    plt.legend()
    plt.savefig('prediction_result.png')
    plt.close()
    
    # Plotly绘图（交互式）
    try:
        # 读取时间戳（如果有）
        df = pd.read_csv('./data/df.csv', sep=',')
        if 'time' in df.columns:
            time_index = df['time'].values
    except:
        pass
    
    fig = go.Figure()
    
    # 添加原始数据线
    fig.add_trace(go.Scatter(
        x=time_index,
        y=timeseries.flatten(),
        name='原始数据',
        line=dict(color='blue')
    ))
    
    # 添加训练集预测线
    fig.add_trace(go.Scatter(
        x=time_index,
        y=train_plot.flatten(),
        name='训练集预测',
        line=dict(color='red')
    ))
    
    # 添加验证集预测线
    fig.add_trace(go.Scatter(
        x=time_index,
        y=val_plot.flatten(),
        name='验证集预测',
        line=dict(color='yellow')
    ))
    
    # 添加测试集预测线
    fig.add_trace(go.Scatter(
        x=time_index,
        y=test_plot.flatten(),
        name='测试集预测',
        line=dict(color='green')
    ))
    
    # 更新布局
    fig.update_layout(
        title='LSTM时序预测结果',
        xaxis_title='时间',
        yaxis_title='价差',
        hovermode='x unified',
        template='plotly_white',
        xaxis=dict(
            rangeslider=dict(visible=True),
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1小时", step="hour", stepmode="backward"),
                    dict(count=1, label="1天", step="day", stepmode="backward"),
                    dict(count=7, label="1周", step="day", stepmode="backward"),
                    dict(step="all", label="全部")
                ])
            )
        )
    )
    
    # 保存为HTML文件
    fig.write_html('prediction_result.html')
    
    # 打印最后一个预测值
    last_prediction = test_plot[-1]
    print(f"最后预测值: {last_prediction[0]:.4f}")

if __name__ == "__main__":
    main() 