import plotly.graph_objects as go
import numpy as np
import torch
import pandas as pd
from lstm_demo import model, X_train, X_test, timeseries, train_size, lookback

# 读取原始数据以获取时间戳
df = pd.read_csv('./data/df.csv', sep=',')
time_stamps = df['time'].values

# 确保模型处于评估模式
model.eval()

# 生成预测
with torch.no_grad():
    # 训练集预测
    train_plot = np.ones_like(timeseries) * np.nan
    train_predictions = model(X_train).cpu().numpy()  # shape: [batch_size, seq_len, 1, 1]
    train_predictions = train_predictions.squeeze(-1)  # shape: [batch_size, seq_len, 1]
    train_plot[lookback:train_size] = train_predictions[:, -1, 0]  # 只取最后一个时间步的预测
    
    # 测试集预测
    test_plot = np.ones_like(timeseries) * np.nan
    test_predictions = model(X_test).cpu().numpy()  # shape: [batch_size, seq_len, 1, 1]
    test_predictions = test_predictions.squeeze(-1)  # shape: [batch_size, seq_len, 1]
    test_plot[train_size+lookback:len(timeseries)] = test_predictions[:, -1, 0]  # 只取最后一个时间步的预测

# 创建交互式图表
fig = go.Figure()

# 添加原始数据线
fig.add_trace(go.Scatter(
    x=time_stamps,
    y=timeseries.flatten(),
    name='原始数据',
    line=dict(color='blue')
))

# 添加训练集预测线
fig.add_trace(go.Scatter(
    x=time_stamps,
    y=train_plot.flatten(),
    name='训练集预测',
    line=dict(color='red')
))

# 添加测试集预测线
fig.add_trace(go.Scatter(
    x=time_stamps,
    y=test_plot.flatten(),
    name='测试集预测',
    line=dict(color='green')
))

# 更新布局
fig.update_layout(
    title='LSTM时序预测结果',
    xaxis_title='时间',
    yaxis_title='价差',
    hovermode='x unified',  # 启用统一的悬停模式
    template='plotly_white',  # 使用白色主题
    xaxis=dict(
        rangeslider=dict(visible=True),  # 添加范围滑块
        rangeselector=dict(  # 添加时间范围选择器
            buttons=list([
                dict(count=1, label="1小时", step="hour", stepmode="backward"),
                dict(count=1, label="1天", step="day", stepmode="backward"),
                dict(count=7, label="1周", step="day", stepmode="backward"),
                dict(step="all", label="全部")
            ])
        )
    )
)

# 显示图表
fig.show()

# 保存为HTML文件
fig.write_html('timeseries_plot.html')

# 打印最后一个预测值
last_prediction = test_plot[-1]
last_time = time_stamps[-1]
print(f"最后预测时间: {last_time}")
print(f"最后预测值: {last_prediction:.4f}") 