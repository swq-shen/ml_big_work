import os  # 导入os模块，用于操作文件和目录路径
import pandas as pd  # 导入pandas模块，用于数据处理和分析
from tradingcalendar import Calendar  # 导入Calendar类，用于判断是否为交易日

class DataLoader(object):  # 定义MeowDataLoader类
    def __init__(self, h5dir):  # 初始化方法，接收h5文件存储目录作为参数
        self.h5dir = h5dir  # 存储h5文件目录的属性
        self.calendar = Calendar()  # 初始化Calendar对象，用于判断日期是否为交易日

    def loadDates(self, dates):  # 接收日期列表，加载这些日期的数据
        if len(dates) == 0:  # 如果日期列表为空，抛出ValueError异常
            raise ValueError("Dates empty")
        # 使用列表推导式和pd.concat合并每个日期的数据
        return pd.concat(self.loadDate(x) for x in dates)

    def loadDate(self, date):  # 接收单个日期，加载该日期的数据
        if not self.calendar.isTradingDay(date):  # 如果日期不是交易日，抛出ValueError异常
            raise ValueError("Not a trading day: {}".format(date))
        # 构造.h5文件的完整路径
        h5File = os.path.join(self.h5dir, "{}.h5".format(date))
        # 使用pandas的read_hdf方法读取.h5文件
        df = pd.read_hdf(h5File)
        # 将日期添加到DataFrame的列中
        df.loc[:, "date"] = date
        # 预定义的列名，用于重新排列DataFrame的列
        precols = ["symbol", "interval", "date"]
        # 重新排列列，将预定义的列放在前面，其余列按原顺序排列在后
        df = df[precols + [x for x in df.columns if x not in precols]]
        return df  # 返回处理后的DataFrame
