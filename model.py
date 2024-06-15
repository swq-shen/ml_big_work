import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from dl import DataLoader
from eval import MeowEvaluator
from feat import FeatureGenerator
from log import log  # 假设您的日志模块与之前相同
from tradingcalendar import Calendar
from eval import show_
from eval import composite_loss
from torch.optim.lr_scheduler import StepLR


class ResNLS(nn.Module):
    def __init__(self, input_channels, hidden_size=16, output_size=1):
        super(ResNLS, self).__init__()
        self.cnn1 = nn.Conv1d(in_channels=input_channels,
                              out_channels=hidden_size,
                              kernel_size=4, stride=1, padding=1)
        self.cnn2 = nn.Conv1d(in_channels=hidden_size,
                              out_channels=hidden_size,
                              kernel_size=4, stride=1, padding=1)

        self.cnn3 = nn.Conv1d(in_channels=hidden_size,
                              out_channels=hidden_size,
                              kernel_size=4, stride=1, padding=1)

        self.dropout = nn.Dropout(p=0.5)  # 添加Dropout层
        self.lstm = nn.LSTM(input_size=hidden_size * 3,
                            hidden_size=hidden_size,
                            num_layers=4,
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn1(x)
        x = F.relu(x)
        x = self.cnn2(x)
        x = F.relu(x)
        x = self.cnn3(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = torch.cat((x, x), dim=1)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)  # LSTM层
        x = x[:, -1, :]  # 取序列的最后一个时间步
        x = self.dropout(x)
        x = self.linear(x)  # 线性层
        return x


class Model(object):
    def __init__(self, device, model=None):
        self.device = device
        if model is None:
            # 假设输入通道数为1，隐藏层大小为128，输出大小为1
            self.model = ResNLS(input_channels=1, hidden_size=32, output_size=1).to(device)
        else:
            self.model = model.to(device)
        self.evaluator = MeowEvaluator("../archive")  # 初始化评估器
        self.calendar = Calendar()  # 初始化交易日历
        self.dloader = DataLoader("../archive")  # 初始化数据加载器
        self.featGenerator = FeatureGenerator("../archive")
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
        # self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.5)

    def fit(self, xdf, ydf, epochs=1, batch_size=64):
        t_dates = self.calendar.range(20231201, 20231229)  # 获取交易日历的日期范围
        t_rawData = self.dloader.loadDates(t_dates)  # 加载指定日期范围内的原始数据
        t_xdf, t_ydf = self.featGenerator.genFeatures(t_rawData)  # 生成特征和目标数据
        # 转换数据到张量

        t_xdf = torch.tensor(t_xdf.values, dtype=torch.float32).to(self.device)

        # 转换数据到张量
        xdf = torch.tensor(xdf.values, dtype=torch.float32).to(self.device)  # xdf的形状是(batch_size, num_features)
        ydf = torch.tensor(ydf.values, dtype=torch.float32).to(self.device)  # ydf的形状是(batch_size,)
        # 初始化优化器和损失函数
        criterion = composite_loss
        # 将模型设置为训练模式
        self.model.train()
        # 计算批次数量
        num_batches = xdf.size(0) // batch_size
        # 训练循环
        for epoch in range(epochs):
            print(epoch)
            for batch in range(num_batches):
                # 创建数据的批次
                start_idx = batch * batch_size
                end_idx = start_idx + batch_size
                batch_x = xdf[start_idx:end_idx]
                batch_y = ydf[start_idx:end_idx]
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                p, r, e, y = self.tes(t_xdf, t_ydf)
                lr = lr_upd(p, r, e)
                # for param_group in self.optimizer.param_groups:
                #     param_group['lr'] = lr
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}")

        print("Training complete")

    def predict(self, xdf, batch_size=2048):
        # 将模型设置为评估模式
        self.model.eval()
        with torch.no_grad():
            # 初始化一个列表来存储每个批次的预测结果
            predictions_list = []
            # 计算需要的批次数量，包括最后一个不完整的批次
            num_batches = (len(xdf) + batch_size - 1) // batch_size
            # 分批次进行预测
            for batch in range(num_batches):
                # 定义当前批次的起始和结束索引
                start_idx = batch * batch_size
                end_idx = min(start_idx + batch_size, len(xdf))  # 确保不会超出xdf的长度
                # 获取当前批次的数据
                batch_x = xdf[start_idx:end_idx].to(self.device)
                # 进行预测
                batch_predictions = self.model(batch_x)
                # 将预测结果添加到列表中
                predictions_list.append(batch_predictions.cpu().numpy())
            # 将所有批次的预测结果合并为一个 NumPy 数组
            predictions = np.concatenate(predictions_list, axis=0)
        return predictions

    def tes(self, xdf, ydf):

        predictions = self.predict(xdf)

        # 确保 ydf 是一个 DataFrame
        if "forecast" not in ydf.columns:
            ydf["forecast"] = pd.Series(index=ydf.index)  # 初始化空列

        # 将预测结果添加到 ydf 的 "forecast" 列中
        # 确保预测结果的长度与 ydf 的行数相匹配
        ydf["forecast"] = predictions
        p, r, e = self.evaluator.eval(ydf)
        return p, r, e, ydf
    def save_net(self, path):
        # ... 保存模型的代码 ...
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
        torch.save(checkpoint, path)

    def load_net(self, path):
        # ... 加载模型的代码 ...
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])


def lr_upd(p, r, e):
    if p <= 0 and r < 0:lr = 0.01
    elif p <=0:lr =0.005
    elif r <= 0:lr=0.005
    else:lr = 0.0001
    return lr
