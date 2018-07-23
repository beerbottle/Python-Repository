import numpy as np
import pandas as pd


def inner_chi2(p_df, p_total_col, p_bad_col, p_overall_rate):
    """
    内部函数：计算卡方
    :param p_df:
    :param p_total_col:
    :param p_bad_col:
    :param p_overall_rate:
    :return: 卡方结果
    """
    df2 = p_df.copy()
    df2['expected'] = p_df[p_total_col].apply(lambda x: x * p_overall_rate)
    combined = zip(df2['expected'], df2[p_bad_col])
    chi = [(i[0]-i[1])**2/i[0] for i in combined]

    return_chi2 = sum(chi)
    return return_chi2


def inner_assign_group(p_value, p_bin):
    """
    内部函数：分组 将连续变量进行分箱 一般多用于初次分箱 一般默认100箱
    :param p_value:
    :param p_bin:
    :return: 内部分箱结果
    """
    n = len(p_bin)
    if p_value <= min(p_bin):
        return min(p_bin)
    elif p_value > max(p_bin):
        return 10e10
    else:
        for i in range(n-1):
            if p_bin[i] < p_value <= p_bin[i + 1]:
                return p_bin[i + 1]


def inner_assign_bin(x, p_cutoff_points):
    """
    根据分箱边界值进行变量分箱
    :param x: 变量series
    :param p_cutoff_points:
    :return: 返回分箱结果
    """
    """
    :param x: the value of variable
    :param p_cutoff_points: the ChiMerge result for continous variable
    :return: bin number, indexing from 0
    for example, if cutOffPoints = [10,20,30], if x = 7, return Bin 0. If x = 35, return Bin 3
    """
    num_bin = len(p_cutoff_points) + 1
    if x <= p_cutoff_points[0]:
        return 'Bin 0'
    elif x > p_cutoff_points[-1]:
        return 'Bin {}'.format(num_bin-1)
    else:
        for i in range(0, num_bin-1):
            if p_cutoff_points[i] < x <= p_cutoff_points[i + 1]:
                return 'Bin {}'.format(i+1)


def inner_chi_merge_max_interval(p_df, p_col, p_target, p_max_bin=5, special_attribute=[]):
    """
    进行卡方分箱
    :param p_df:
    :param p_col: 要进行卡方分箱的col
    :param p_target: 目标col
    :param p_max_bin: 最大分箱数
    :param special_attribute: 特殊的分箱要求 这是没有使用
    :return: 返回卡方分箱的list
    """

    # p_df = pd1
    # p_col = "day"
    # p_target = "chrun"
    # p_max_bin = 5

    col_level = sorted(list(set(p_df[p_col])))
    if len(col_level) <= p_max_bin:
        print("The number of original levels for {} is less than or equal to max intervals".format(p_col))
        return []

    temp_df2 = p_df.copy()
    n_distinct = len(col_level)
    if n_distinct > 100:
        ind_x = [int(i/100.00 * n_distinct) for i in range(1, 100)]
        split_x = [col_level[i] for i in ind_x]
        temp_df2["temp"] = temp_df2[p_col].map(lambda x: inner_assign_group(x, split_x))
    else:
        temp_df2['temp'] = p_df[p_col]

    total = temp_df2.groupby(['temp'])[p_target].count()
    total = pd.DataFrame({'total': total})
    bad = temp_df2.groupby(['temp'])[p_target].sum()
    bad = pd.DataFrame({'bad': bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    n = sum(regroup['total'])
    b = sum(regroup['bad'])
    # the overall bad rate will be used in calculating expected bad count
    overall_rate = b * 1.0 / n

    col_level = sorted(list(set(temp_df2['temp'])))
    group_intervals = [[i] for i in col_level]
    group_num = len(group_intervals)

    while len(group_intervals) > p_max_bin:
        chisq_list = []
        for interval in group_intervals:
            df2b = regroup.loc[regroup['temp'].isin(interval)]
            chisq = inner_chi2(df2b, 'total', 'bad', overall_rate)
            chisq_list.append(chisq)
        min_position = chisq_list.index(min(chisq_list))
        if min_position == 0:
            combined_position = 1
        elif min_position == group_num - 1:
            combined_position = min_position - 1
        else:
            if chisq_list[min_position - 1] <= chisq_list[min_position + 1]:
                combined_position = min_position - 1
            else:
                combined_position = min_position + 1
        group_intervals[min_position] = group_intervals[min_position] + group_intervals[combined_position]
        group_intervals.remove(group_intervals[combined_position])
        group_num = len(group_intervals)

    group_intervals = [sorted(i) for i in group_intervals]
    cut_off_points = [max(i) for i in group_intervals[:-1]]
    cut_off_points = special_attribute + cut_off_points
    return cut_off_points


def bin_chi2(p_df, p_num_var_list, p_target, p_max_bin=5):
    """
    批量将数据集数字型特征进行卡方分箱 默认5箱
    :param p_df:
    :param p_num_var_list:
    :param p_target:
    :param p_max_bin:
    :return:
    """
    for var in p_num_var_list:
        cutoff_points = inner_chi_merge_max_interval(p_df, var, p_target, p_max_bin)
        var_cutoff = {}
        var_cutoff[var] = cutoff_points
        p_df[var] = p_df[var].map(lambda x: inner_assign_bin(x, cutoff_points))
    print("function bin_chi2 finished!...................")


def calc_woe(p_df, p_col, p_target):
    """
    计算WOE
    :param p_df:
    :param p_col:
    :param p_target:
    :return:
    """
    return

def bin_best_ks(p_df, p_var, p_target):
    """
    ks分箱法 不推荐
    :param p_df:
    :param p_var:
    :param p_target:
    :return:
    """
    return





