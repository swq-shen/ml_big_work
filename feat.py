


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




