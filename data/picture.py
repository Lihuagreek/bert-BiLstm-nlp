

# df = pd.Series(np.random.randn(100), index=pd.date_range('1/1/2020', periods=100))
# df = df.cumsum()
# df.plot()
# plt.show()
# df = pd.DataFrame(np.random.randn(1000, 3), index=pd.date_range('1/1/2020', periods=1000), columns=['A','B','C'])
# df = df.cumsum()
# df.plot()
# plt.show()
# !/user/bin/env python
# coding=utf-8

# import pandas as pd
# from matplotlib import pyplot as plt
#
# _dates = ['1','2','3','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1']
# _data1 = [1, 2, 4, 6, 3, 2, 5, 7, 8, 0]
# _data2 = [0, 9, 8, 2, 1, 0, 6, 5, 2, 1]
#
# di = pd.DatetimeIndex(_dates,
#                       dtype='datetime64[ns]', freq=None)
#
# pd.DataFrame({'data1': _data1},
#              index=di).plot.line()  # 图形横坐标默认为数据索引index。
# #
# # plt.savefig(r'data/p1.png', dpi=200)
# plt.show()  # 显示当前正在编译的图像
#
# pd.DataFrame({'data1': _data1, 'data2': _data2},
#              index=di).plot.line()  # 图形横坐标默认为数据索引index。
# #
# # plt.savefig(r'data/p2.png', dpi=200)
# plt.show()  # 显示当前正在编译的图像


import pandas as pd
import matplotlib.pyplot as plt


# def column_chart(excel_path, sheet_name):
#     """
#     柱状图
#     :param excel_path:
#     :param sheet_name:
#     :return:
#     """
#     plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
#     plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
#     df = pd.read_excel(excel_path, sheet_name=sheet_name)
#     print(df)
#     # 接下来我们通过pandas库下面的bar来设置柱形图的X，Y坐标轴
#     df.plot.bar(x="月份", y="页面浏览量（PV）")
#     # 然后通过pyplot的show方法将柱形图进行展示出来
#     plt.show()


def line_chart():
    """
    折线图
    :return:
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    df = pd.read_excel("example.xls", sheet_name="Sheet1")
    plt.xlabel('迭代次数')
    plt.ylabel('F1 值')
    x = df["迭代次数"]
    y1 = df["朴素贝叶斯"]
    y2 = df["SVM"]
    y3 = df["BiLSTM"]
    y4 = df["BERT"]
    y5 = df["向量拼接BERT-BiLSTM模型"]
    # df.plot(y=['朴素贝叶斯','SVM',"BiLSTM","BERT","向量拼接BERT-BiLSTM模型"],marker ='.')
    plt.plot(x, y1,marker ='.',linestyle='--',label='朴素贝叶斯')
    plt.plot(x, y2,marker ='+',linestyle='-',label='SVM')
    plt.plot(x, y3,marker ='*',linestyle='-',label='BiLSTM')
    plt.plot(x, y4,marker ='.',linestyle='-',label='BERT')
    plt.plot(x, y5,label='向量拼接BERT-BiLSTM模型')
    # plt.xlabel("月份")
    # plt.ylabel("PV")
    plt.savefig(r'data/p1.png', dpi=200)
    plt.legend()
    plt.show()
def line_chart2():
    """
    折线图
    :return:
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    df = pd.read_excel("example.xls", sheet_name="Sheet2")
    plt.xlabel('迭代次数')
    plt.ylabel('训练与验证acc数值')
    x = df["迭代次数"]
    y1 = df["acc"]
    y2 = df["val_acc"]
    # df.plot(y=['朴素贝叶斯','SVM',"BiLSTM","BERT","向量拼接BERT-BiLSTM模型"],marker ='.')
    plt.plot(x, y1,marker ='.',linestyle='--',label='acc')
    plt.plot(x, y2,marker ='+',linestyle='-',label='val_acc')
    plt.savefig(r'data/p1.png', dpi=200)
    # plt.xlabel("月份")
    # plt.ylabel("PV")

    plt.legend()
    plt.show()
def line_chart3():
    """
    折线图
    :return:
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    df = pd.read_excel("example.xls", sheet_name="Sheet3")
    plt.xlabel('迭代次数')
    plt.ylabel('训练与验证loss数值')
    x = df["迭代次数"]
    y1 = df["loss"]
    y2 = df["val_loss"]
    # df.plot(y=['朴素贝叶斯','SVM',"BiLSTM","BERT","向量拼接BERT-BiLSTM模型"],marker ='.')
    plt.plot(x, y1,marker ='.',linestyle='--',label='loss')
    plt.plot(x, y2,marker ='+',linestyle='-',label='val_loss')
    plt.savefig(r'data/p2.png', dpi=200)
    # plt.xlabel("月份")
    # plt.ylabel("PV")

    plt.legend()
    plt.show()

# def pie_chart():
#     plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
#     plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
#     df = pd.read_excel("example.xlsx", sheet_name="Sheet2")
#     labels = ["1月", "2月", "3月", "4月", "5月", "6月", "7月", "8月", "9月", "10月", "11月", "12月"]
#     x = df["贡献下游浏览量"]
#     plt.pie(x, labels=labels,  autopct='%1.1f%%')
#     plt.axis("equal")
#     plt.title('贡献下游浏览量')
#     plt.show()


if __name__ == '__main__':
    # line_chart()
    line_chart2()
    line_chart3()
