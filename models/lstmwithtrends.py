import torch
import torch.nn as nn
import numpy as np

class DeepTrendDecomposer(nn.Module):
    def __init__(self, window_size=5):
        super().__init__()
        # 使用大窗口卷积捕获低频趋势
        self.trend_extractor = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=window_size, padding='same'),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=window_size, padding='same')
        )
        
    def forward(self, x):
        # 输入形状: (batch_size, seq_len)
        x = x.permute(0,2,1)  # 添加通道维度 -> (B, 1, T)
        # print(x.shape)
        trend = self.trend_extractor(x)  # (B, 1, T)
        residual = x - trend
        return trend, residual

# # 使用示例
# model = DeepTrendDecomposer(window_size=7)
# input_seq = torch.randn(32, 100)  # batch_size=32, seq_len=100
# trend, residual = model(input_seq)


class TrendAwareLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50):
        super().__init__()
        self.decomposer =  DeepTrendDecomposer(window_size=7)
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, 1)


    def forward(self, x):
        # 分解趋势
        trend, residual = self.decomposer(x)
        # print(trend.shape,residual.shape)
        
        # 用LSTM预测残差
        trend = trend.permute(0,2,1)
        residual = residual.permute(0,2,1)
        # residual = residual.unsqueeze(-1)  # (B, T, 1)
        lstm_out, _ = self.lstm(residual)
        pred_residual = self.linear(lstm_out[:, -1, :])
        
        # 预测趋势（简单移动平均）
        pred_trend = trend[:, -1].unsqueeze(-1)  # 假设趋势缓慢变化
        
        return pred_trend + pred_residual
# model = TrendAwareLSTM().to(device)
# model 

# sum(para.numel() for para in model.parameters())