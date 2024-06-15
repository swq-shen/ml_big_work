
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt


class FeatureGenerator(object):  # 定义MeowFeatureGenerator类

    @classmethod  # 类方法，不需要实例化类即可调用
    def featureNames(cls):  # 返回特征名称列表
        return [
            "ob_imb0",  # 特征名称：订单簿不平衡0
            "ob_imb4",  # 特征名称：订单簿不平衡4
            "ob_imb9",  # 特征名称：订单簿不平衡9
            "trade_imb",  # 特征名称：交易不平衡
            "trade_imbema5",  # 特征名称：交易不平衡指数移动平均5
            "lagret12",  # 特征名称：滞后回报12
        # ]
        # return [
        #     "best_bid_change",  # 最佳买价变动：买一档价格的变化量
        #     "best_ask_change",  # 最佳卖价变动：卖一档价格的变化量
        #     "buy_depth_std",  # 订单簿买深度标准差：衡量买方订单簿深度的分散程度
        #     "sell_depth_std",  # 订单簿卖深度标准差：衡量卖方订单簿深度的分散程度
        #     "buy_depth_skew",  # 订单簿买深度偏度：衡量买方订单簿深度分布的偏斜程度
        #     "sell_depth_skew",  # 订单簿卖深度偏度：衡量卖方订单簿深度分布的偏斜程度
        #     "buy_depth_kurt",  # 订单簿买深度峰度：衡量买方订单簿深度分布的尖峭程度
        #     "sell_depth_kurt",  # 订单簿卖深度峰度：衡量卖方订单簿深度分布的尖峭程度

            # "turnover_to_volume_ratio",  # 买卖交易额与成交量的比率：交易额相对于成交量的比例
            # "best_bid_ask_spread",  # 最高买单价与最低卖单价的差额：买卖价差的直接度量
            # "best_bid_to_mid_ratio",  # 最高买单价与中间价的比率：买一档价格相对于中间价的比例
            # "best_ask_to_mid_ratio",  # 最低卖单价与中间价的比率：卖一档价格相对于中间价的比例
            # "midpx_moving_change_rate",  # 中间价的移动变化率：中间价移动平均与前一期中间价的变化率
            # "volume_moving_change_rate",  # 成交量的移动变化率：成交量移动平均与前一期成交量的变化率
            # "add_cxl_order_turnover_ratio",  # 新增订单额与取消订单额的比率
            # "trade_count_rolling_mean",  # 买卖交易次数的移动平均：交易次数的近期平均
        ]

    def __init__(self, cacheDir):  # 初始化方法，接收缓存目录作为参数
        self.cacheDir = cacheDir  # 存储缓存目录的属性
        self.ycol = "fret12"  # 目标列的名称
        self.mcols = ["symbol", "date", "interval"]  # 合并列的名称

    def genFeatures(self, df):  # 定义生成特征的方法，接收DataFrame

        # 计算特征值，这里列出了特征的计算方式：
        # 订单簿不平衡特征，使用买卖订单量差额除以总和
        df.loc[:, "ob_imb0"] = (df["asize0"] - df["bsize0"]) / (df["asize0"] + df["bsize0"])
        df.loc[:, "ob_imb4"] = (df["asize0_4"] - df["bsize0_4"]) / (df["asize0_4"] + df["bsize0_4"])
        df.loc[:, "ob_imb9"] = (df["asize5_9"] - df["bsize5_9"]) / (df["asize5_9"] + df["bsize5_9"])

        # 交易不平衡特征，使用买入交易量与卖出交易量的差额除以总和
        df.loc[:, "trade_imb"] = (df["tradeBuyQty"] - df["tradeSellQty"]) / (df["tradeBuyQty"] + df["tradeSellQty"])

        # 交易不平衡指数移动平均5，使用ewm方法计算半衰期为5的指数移动平均
        df.loc[:, "trade_imbema5"] = df["trade_imb"].ewm(halflife=5).mean()

        # 计算滞后回报12，即过去12个时间单位的回报率
        df.loc[:, "bret12"] = (df["midpx"] - df["midpx"].shift(12)) / df["midpx"].shift(12)

        # 计算交叉间隔的滞后回报平均值，并将其与原始滞后回报合并
        cxbret = df.groupby("interval")[["bret12"]].mean().reset_index().rename(columns={"bret12": "cx_bret12"})
        df = df.merge(cxbret, on="interval", how="left")

        # 计算滞后回报12的差值
        df.loc[:, "lagret12"] = df["bret12"] - df["cx_bret12"]

        # 将特征和目标列设置为DataFrame的索引，并填充缺失值
        xdf = df[self.mcols + self.featureNames()].set_index(self.mcols).fillna(0)
        ydf = df[self.mcols + [self.ycol]].set_index(self.mcols).fillna(0)

        return xdf, ydf  # 返回特征DataFrame和目标DataFrame
    #
    # def genFeatures(self, df):
    #     # 最佳买价变动
    #     df['best_bid_change'] = df['bid0'] - df['bid0'].shift(1)  # 买一档价格的变化量
    #     # 最佳卖价变动
    #     df['best_ask_change'] = df['ask0'] - df['ask0'].shift(1)  # 卖一档价格的变化量
    #     # 订单簿买深度标准差 - 衡量买方订单簿深度的分散程度
    #     # 选取买方订单簿的深度列进行计算
    #     buy_depth_columns = ['bsize0', 'bsize0_4', 'bsize5_9', 'bsize10_19']
    #     df['buy_depth_std'] = df[buy_depth_columns].std(axis=1, skipna=True)
    #
    #     # 订单簿卖深度标准差 - 衡量卖方订单簿深度的分散程度
    #     # 选取卖方订单簿的深度列进行计算
    #     sell_depth_columns = ['asize0', 'asize0_4', 'asize5_9', 'asize10_19']
    #     df['sell_depth_std'] = df[sell_depth_columns].std(axis=1, skipna=True)
    #
    #     # 订单簿买深度偏度 - 衡量买方订单簿深度分布的偏斜程度
    #     df['buy_depth_skew'] = df[buy_depth_columns].skew(axis=1, skipna=True)
    #
    #     # 订单簿卖深度偏度 - 衡量卖方订单簿深度分布的偏斜程度
    #     df['sell_depth_skew'] = df[sell_depth_columns].skew(axis=1, skipna=True)
    #
    #     # 订单簿买深度峰度 - 衡量买方订单簿深度分布的尖峭程度
    #     df['buy_depth_kurt'] = df[buy_depth_columns].kurtosis(axis=1, skipna=True)
    #
    #     # 订单簿卖深度峰度 - 衡量卖方订单簿深度分布的尖峭程度
    #     df['sell_depth_kurt'] = df[sell_depth_columns].kurtosis(axis=1, skipna=True)
    #
    #     # 计算买卖价差
    #     df['bid_ask_spread'] = df['ask0'] - df['bid0']
    #
    #
    #     # 成交量与订单簿深度的比率 - 成交量相对于订单簿深度的比例
    #     df['volume_to_depth_ratio'] = (df['tradeBuyQty'] + df['tradeSellQty']) / (df['bsize0'] + df['asize0'])
    #
    #     # 新增买单量与取消买单量的比率
    #     df['add_cancel_buy_ratio'] = df['nAddBuy'] / df['nCxlBuy']
    #     # 新增卖单量与取消卖单量的比率
    #     df['add_cancel_sell_ratio'] = df['nAddSell'] / df['nCxlSell']
    #
    #     # 计算买卖交易额
    #     df['trade_turnover'] = df['tradeBuyTurnover'] + df['tradeSellTurnover']
    #     # 买卖交易额与成交量的比率
    #     df['turnover_to_volume_ratio'] = df['trade_turnover'] / (df['tradeBuyQty'] + df['tradeSellQty'])
    #
    #     # 最高买单价与最低卖单价的差额
    #     df['best_bid_ask_spread'] = df['ask0'] - df['bid0']
    #
    #     # 最高买单价与中间价的比率
    #     df['best_bid_to_mid_ratio'] = df['bid0'] / df['midpx']
    #     # 最低卖单价与中间价的比率
    #     df['best_ask_to_mid_ratio'] = df['ask0'] / df['midpx']
    #
    #     # 中间价的移动平均
    #     df['midpx_rolling_mean'] = df['midpx'].rolling(window=5).mean()
    #     # 中间价的移动变化率 - 中间价移动平均与前一期中间价的变化率
    #     df['midpx_moving_change_rate'] = (df['midpx_rolling_mean'] - df['midpx'].shift(1)) / df['midpx'].shift(1)
    #
    #     # 成交量的移动平均
    #     df['volume_rolling_mean'] = (df['tradeBuyQty'] + df['tradeSellQty']).rolling(window=5).mean()
    #     # 成交量的移动变化率 - 成交量移动平均与前一期成交量的变化率
    #     df['volume_moving_change_rate'] = (df['volume_rolling_mean'] - (df['tradeBuyQty'] + df['tradeSellQty']).shift(
    #         1)) / (df['tradeBuyQty'] + df['tradeSellQty']).shift(1)
    #
    #     # 新增订单额与取消订单额的比率
    #     df['add_cxl_order_turnover_ratio'] = (df['addBuyTurnover'] + df['addSellTurnover']) / (
    #                 df['cxlBuyTurnover'] + df['cxlSellTurnover'])
    #
    #     # 买卖交易次数的移动平均
    #     df['trade_count_rolling_mean'] = (df['nTradeBuy'] + df['nTradeSell']).rolling(window=5).mean()
    #
    #
    #
    #     xdf = df[self.mcols + self.featureNames()].set_index(self.mcols)
    #     ydf = df[self.mcols + [self.ycol]].set_index(self.mcols)
    #
    #
    #     return xdf, ydf  # 返回特征DataFrame和目标DataFrame


    def genAll(self, df):

            # 定义所有的列
        x = [
            'midpx', 'lastpx', 'open',
       'high', 'low', 'bid0', 'ask0', 'bid4', 'ask4', 'bid9', 'ask9', 'bid19',
       'ask19', 'bsize0', 'asize0', 'bsize0_4', 'asize0_4', 'bsize5_9',
       'asize5_9', 'bsize10_19', 'asize10_19', 'btr0_4', 'atr0_4', 'btr5_9',
       'atr5_9', 'btr10_19', 'atr10_19', 'nTradeBuy', 'tradeBuyQty',
       'tradeBuyTurnover', 'tradeBuyHigh', 'tradeBuyLow', 'buyVwad',
       'nTradeSell', 'tradeSellQty', 'tradeSellTurnover', 'tradeSellHigh',
       'tradeSellLow', 'sellVwad', 'nAddBuy', 'addBuyQty', 'addBuyTurnover',
       'addBuyHigh', 'addBuyLow', 'nAddSell', 'addSellQty', 'addSellTurnover',
       'addSellHigh', 'addSellLow', 'nCxlBuy', 'cxlBuyQty', 'cxlBuyTurnover',
       'cxlBuyHigh', 'cxlBuyLow', 'nCxlSell', 'cxlSellQty', 'cxlSellTurnover',
       'cxlSellHigh', 'cxlSellLow']

        xdf = df[self.mcols + x].set_index(self.mcols).fillna()
        ydf = df[self.mcols + [self.ycol]].set_index(self.mcols).fillna(0)

        return xdf, ydf

    @classmethod
    def plot_correlation_heatmap(self, df):
        # 计算特征列之间的相关系数矩阵
        corr_matrix = df.corr()

        # 使用Seaborn绘制热力图
        plt.figure(figsize=(10, 8))  # 可以根据需要调整图形大小
        sns.heatmap(corr_matrix, annot=False, fmt=".2f", cmap='coolwarm', square=True)
        plt.title('Feature Correlation Heatmap')
        plt.show()

