import pandas as pd
import numpy as np
from scipy.interpolate import lagrange
# import matplotlib.pyplot as plt
# from pylab import mpl
# import math  # used in Pycharm
# # %matplotlib inline  # used for jupyter notebook


def _get_diff_table(X, Y):
    """
    得到插商表
    """
    n = len(X)
    A = np.zeros([n, n])

    for i in range(0, n):
        A[i][0] = Y[i]

    for j in range(1, n):
        for i in range(j, n):
            A[i][j] = (A[i][j - 1] - A[i - 1][j - 1]) / (X[i] - X[i - j])

    return A


def _newton_interpolation(X, Y, x):
    """
    计算x点的插值
    """
    sum = Y[0]
    temp = np.zeros((len(X), len(X)))
    # 将第一行赋值
    for i in range(0, len(X)):
        temp[i, 0] = Y[i]
    temp_sum = 1.0
    for i in range(1, len(X)):
        # x的多项式
        temp_sum=temp_sum*(x-X[i-1])
        # 计算均差
        for j in range(i, len(X)):
            temp[j, i] = (temp[j, i-1] - temp[j-1, i-1]) / (X[j] - X[j-i])
        sum += temp_sum * temp[i, i]
    return sum


def _get_newton_value(T, SF, num):
    sfa = _get_diff_table(T, SF)
    df = pd.DataFrame(sfa)
    # print(df)
    xsf = np.linspace(np.min(T), np.max(T), num, endpoint=True)
    ysf = []
    for x in xsf:
        ysf.append(_newton_interpolation(T, SF, x))
    # print(ysf)
    return ysf


def _split_file_10T():
    data_60 = 'E:/5 - Work/7 - hx/data_process/CLINKER 6O IBM.xlsx'
    df_data_60 = pd.read_excel(data_60)

    # 把时间对齐
    df_data_60['Sample Date'] = df_data_60['Sample Date'].dt.floor('10T')
    df_data_60 = df_data_60.dropna(axis=0, how='any')
    df_data_60.index = pd.to_datetime(df_data_60['Sample Date'])
    columns_clean = df_data_60.describe().dropna(how='any', axis=1).columns
    # print(columns_clean)
    df_data_60_clean = df_data_60[columns_clean].dropna(axis=0, how='any')
    # print(df_data_60_clean)
    print(df_data_60_clean.index)

    # 获得文件中每行的时间间隔
    prev_time = df_data_60_clean.index[0]
    # print(prev_time)
    gaps = []
    hours = []
    for time in df_data_60_clean.index[1:]:
        gap = pd.Timedelta(time - prev_time).seconds / 3600.0 + pd.Timedelta(time - prev_time).days * 24
        if gap >= 24:
            hours.append(time - prev_time)
        prev_time = time
        # 对时间进行四舍五入以取整
        gaps.append(int("{:.0f}".format(gap)))
    print(hours)
    print(gaps)

    # 把间隔超过一天的数据进行拆分
    # df_data_60_clean.iloc[1]['chunk'] = 0
    print(df_data_60_clean)
    num = []
    for gap in gaps:
        if gap >= 24:
            num.append([gap, gaps.index(gap)])
    num.append([1, len(df_data_60_clean)])
    # print(num)

    start = 0
    # count = 0
    for i in range(len(num)):
        n = num[i][1]+1
        df = df_data_60_clean[start:n]
        # count += len(df)
        start = n
        fn = 'E:/5 - Work/7 - hx/data_process/CLINKER 6O IBM split_' + str(i) + '.xlsx'
        print(fn)
        df.to_excel(fn, index=True)

    # print(len(gaps))


def _adjust_time(time):
    m = time.minute
    if (m % 10) < 8:
        gap = 10 - 8 + (m % 10)
        # print(gap)
    else:
        gap = (m % 10) - 8
        # print(gap)
    time = time - pd.Timedelta(minutes=gap)
    # print(time)
    return time


def main(data_60):
    # data_60 = 'E:/5 - Work/7 - hx/data_process/CLINKER 6O IBM split 0.xlsx'
    df_data_60 = pd.read_excel(data_60)

    # 把时间对齐
    df_data_60['Sample Date'] = df_data_60['Sample Date'].dt.floor('1T')
    df_data_60 = df_data_60.dropna(axis=0, how='any')
    df_data_60.index = pd.to_datetime(df_data_60['Sample Date'])
    columns_clean = df_data_60.describe().dropna(how='any', axis=1).columns
    # print(columns_clean)
    df_data_60_clean = df_data_60[columns_clean].dropna(axis=0, how='any')
    # print(df_data_60_clean)
    print(df_data_60_clean.index)
    df_data_60_clean = df_data_60_clean.loc[~df_data_60_clean.index.duplicated(keep='first')]

    # 获得文件中每行的时间间隔
    prev_time = df_data_60_clean.index[0]
    # print(prev_time)
    gaps = []
    hours = []
    for time in df_data_60_clean.index[1:]:
        print(pd.Timedelta(time - prev_time))
        # 调整分钟
        time = _adjust_time(time)
        prev_time = _adjust_time(prev_time)
        print(time, prev_time)
        gap = pd.Timedelta(time - prev_time).seconds / 600.0 + pd.Timedelta(time - prev_time).days * 24 * 6
        hours.append(time - prev_time)
        prev_time = time
        # 对时间进行四舍五入以取整
        gaps.append(int("{:.0f}".format(gap)))
    # print(hours)
    print(gaps)

    # # 插入数据
    t = df_data_60_clean.resample('10T', base=8).asfreq()
    sample_times = t.index
    print(sample_times)

    bds, lsfs, mss, mas, c3ss, frees, ss = [df_data_60_clean['No. BD'][0]], [df_data_60_clean['LSF Clinker'][0]], \
                                           [df_data_60_clean['MS'][0]], [df_data_60_clean['MA'][0]], \
                                           [df_data_60_clean['C3S (%)'][0]], [df_data_60_clean['Free lime (%)'][0]], \
                                           [df_data_60_clean['S/ALK_Cl'][0]]
    start_time = df_data_60_clean.index[0]
    for i in range(len(gaps)):
        x = [i, i + 1]
        # 列
        bds.extend(_get_newton_value(x, df_data_60_clean['No. BD'][i:i + 2], gaps[i]+1)[1:])
        lsfs.extend(_get_newton_value(x, df_data_60_clean['LSF Clinker'][i:i + 2], gaps[i]+1)[1:])
        mss.extend(_get_newton_value(x, df_data_60_clean['MS'][i:i + 2], gaps[i]+1)[1:])
        mas.extend(_get_newton_value(x, df_data_60_clean['MA'][i:i + 2], gaps[i]+1)[1:])
        c3ss.extend(_get_newton_value(x, df_data_60_clean['C3S (%)'][i:i + 2], gaps[i]+1)[1:])
        frees.extend(_get_newton_value(x, df_data_60_clean['Free lime (%)'][i:i + 2], gaps[i]+1)[1:])
        ss.extend(_get_newton_value(x, df_data_60_clean['S/ALK_Cl'][i:i + 2], gaps[i]+1)[1:])

    print(len(sample_times), len(bds), len(lsfs), len(mss), len(mas), len(c3ss), len(frees), len(ss))

    new_dict = {'Sample Date': sample_times, 'No. BD': bds, 'LSF Clinker': lsfs, 'MS': mss, 'MA': mas,
                'C3S (%)': c3ss, 'Free lime (%)': frees, 'S/ALK_Cl': ss}
    new_dict_df = pd.DataFrame.from_dict(new_dict)
    print(len(new_dict_df.drop_duplicates()))
    print(new_dict_df)
    # print(new_dict_df.drop_duplicates())

    # # 写入文件
    wf = pd.DataFrame(new_dict_df.drop_duplicates())
    return wf


def _lagrange_interpolation(data_60):
    df_data_60 = pd.read_excel(data_60)

    # 把时间对齐
    df_data_60['Sample Date'] = df_data_60['Sample Date'].dt.floor('1T')
    df_data_60 = df_data_60.dropna(axis=0, how='any')
    df_data_60.index = pd.to_datetime(df_data_60['Sample Date'])
    columns_clean = df_data_60.describe().dropna(how='any', axis=1).columns
    df_data_60_clean = df_data_60[columns_clean].dropna(axis=0, how='any')
    print(df_data_60_clean.index)
    # print(df_data_60_clean)
    # df_data_60_clean.index.drop_duplicates()
    print(df_data_60_clean.index.is_unique)
    df_data_60_clean = df_data_60_clean.loc[~df_data_60_clean.index.duplicated(keep='first')]

    # 按10T进行插值
    df = df_data_60_clean.resample('10T', base=8).asfreq()
    df = df.interpolate()
    print(df)

    return df


if __name__ == "__main__":
    # 拆分文件
    _split_file_10T()

    # # 牛顿插值
    # writer1 = pd.ExcelWriter('E:/5 - Work/7 - hx/data_process/CLINKER 6O IBM split 10T_newton.xlsx')
    # for i in range(9):
    #     filename = 'E:/5 - Work/7 - hx/data_process/CLINKER 6O IBM split ' + str(i) + '.xlsx'
    #     print(filename)
    #     df = main(filename)
    #     df.to_excel(writer1, str(i))
    # writer1.save()

    # 拉格朗日插值
    # writer = pd.ExcelWriter('E:/5 - Work/7 - hx/data_process/CLINKER 6O IBM split 10T_lagrange.xlsx')
    # for i in range(9):
    #     filename = 'E:/5 - Work/7 - hx/data_process/CLINKER 6O IBM split ' + str(i) + '.xlsx'
    #     print(filename)
    #     df = _lagrange_interpolation(filename)
    #     df.to_excel(writer, str(i))
    # writer.save()
