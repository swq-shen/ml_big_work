import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from log import log


class MeowEvaluator(object):
    def __init__(self, cacheDir):
        self.cacheDir = cacheDir
        self.predictionCol = "forecast"
        self.ycol = "fret12"

    def eval(self, ydf):
        ydf = ydf.replace([np.inf, -np.inf], np.nan).fillna(0)
        pcor = ydf[[self.predictionCol, self.ycol]].corr().to_numpy()[0, 1]
        r2 = 1 - ((ydf[self.predictionCol] - ydf[self.ycol]) ** 2).sum() / ydf[self.ycol].var() / ydf.shape[0]
        mse = ((ydf[self.predictionCol] - ydf[self.ycol]) ** 2).sum() / ydf.shape[0]
        log.inf("Meow evaluation summary: Pearson correlation={:.8f}, R2={:.8f}, MSE={:.8f}".format(pcor, r2, mse))

    def plot(self, ydf):
        """
        绘制实际值和预测值的图像。
        参数:
        ydf -- 包含预测列和实际值列的DataFrame。
        """
        # 检查所需的列是否存在于DataFrame中
        if self.predictionCol not in ydf.columns or self.ycol not in ydf.columns:
            log.err("DataFrame must contain both 'forecast' and 'fret12' columns.")
            return
        plt.figure(figsize=(10, 5))  # 设置图形大小
        plt.plot(ydf[self.ycol], label='Actual', color='blue')  # 绘制实际值
        plt.plot(ydf[self.predictionCol], label='Forecast', color='red')  # 绘制预测值
        plt.legend()
        plt.title('Forecast vs Actual')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.grid(True)
        plt.show()
