#!/usr/bin/env python
# -*- coding:utf-8 -*-

import io
import os
import sys
import random
import numbers

import numpy as np
import pandas as pd
from matplotlib import pyplot

import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


def get_list_for_number_str_col(df, col_id, col_target):
    """
    将dataframe中的字段名称分为数值型、字符型两个list返回
    :param df: 数据集
    :param col_id: 主键字段名
    :param col_target: 目标字段名
    :return:str_var_list- 字符型变量列表；numberVarlist- 数值型变量列表
    """
    name_of_col = list(df.columns)
    name_of_col.remove(col_target)
    name_of_col.remove(col_id)
    num_var_list = []
    str_var_list = []

    str_var_list = name_of_col.copy()
    for varName in name_of_col:
        if df[varName].dtypes in (np.int, np.int64, np.uint, np.int32, np.float, np.float64, np.float32, np.double):
            str_var_list.remove(varName)
            num_var_list.append(varName)

    df[str_var_list].head()
    df[num_var_list].head()

    return str_var_list, num_var_list


def list2txt(path, file, p_list):
    """
    将list保存到txt中
    :param path: 路径
    :param file: 文件名
    :param p_list:
    :return: 
    """
    file = open(path + '\\' + file, 'w')
    for var in p_list:
        file.write(var)
        file.write('\n')
    file.close()


def txt2list(file):
    """
    从Txt中读取List
    :param file: Txt文件
    :return: Txt文件中保存的List
    """
    a = open(file)
    lines = a.readlines()
    lists = []  # 直接用一个数组存起来就好了
    for line in lines:
        line = line.strip('\n')
        lists.append(line)
    return lists


def num_var_perf(df, var_list, target_var, path, truncation=False):
    """
    探索数值型变量的分布
    :param df: 数据集
    :param var_list:数值型变量名称列表 List类型
    :param target_var: 响应变量名称
    :param path: 保存图片的位置
    :param truncation=False: 是否对数据做95%盖帽处理 默认不盖帽
    :return:
    """
    for var in var_list:
        valid_df = df.loc[df[var] == df[var]][[var, target_var]]
        rec_perc = 100.0*valid_df.shape[0]/df.shape[0]
        rec_perc_fmt = "%.2f%%" % rec_perc
        desc = valid_df[var].describe()
        value_per50 = '%.2e' % desc['50%']
        value_std = '%.2e' % desc['std']
        value_mean = '%.2e' % desc['mean']
        value_max = '%.2e' % desc['max']
        value_min = '%.2e' % desc['min']
        # 样本权重
        bad_df = valid_df.loc[valid_df[target_var] == 1][var]
        nobad_df = valid_df.loc[valid_df[target_var] == 0][var]
        bad_weight = 100.0*np.ones_like(bad_df)/bad_df.size
        good_weight = 100.0*np.ones_like(nobad_df)/nobad_df.size
        # 是否用95分位数进行盖帽
        if truncation:
            per95 = np.percentile(valid_df[var], 95)
            bad_df = bad_df.map(lambda x: min(x, per95))
            nobad_df = nobad_df.map(lambda x: min(x, per95))
        # 画图
        fig, ax = pyplot.subplots()
        ax.hist(bad_df, weights=bad_weight, alpha=0.3, label='bad')
        ax.hist(nobad_df, weights=good_weight, alpha=0.3, label='noBad')
        title_text = var \
                     + ' \n' \
                     + 'VlidePerc:' + rec_perc_fmt \
                     + ';Mean:' + value_mean \
                     + ';Per50:' + value_per50 \
                     + ';Std:' + value_std \
                     + ';\n' \
                     + 'Max:' + value_max \
                     + ';Min:'+value_min
        ax.set(title=title_text, ylabel='% of dataset in bin')
        ax.margins(0.05)
        ax.set_ylim(bottom=0)
        pyplot.legend(loc='upper right')
        fig_save_path = path+'\\'+str(var)+'.png'
        pyplot.savefig(fig_save_path)
        pyplot.close(1)
        # pyplot.show()


def str_var_pref(df, var_list, target_var, path):
    """
    探索字符型变量的分布
    :param df: 数据集
    :param var_list: 字符型型变量名称列表 List类型
    :param target_var: 响应变量名称
    :param path: 保存图片的位置
    :return:
    """
    for var in var_list:
        valid_df = df.loc[df[var] == df[var]][[var, target_var]]
        rec_perc = 100.0*valid_df.shape[0]/df.shape[0]
        rec_perc_fmt = "%.2f%%" % rec_perc
        dict_freq = {}
        dict_bad_rate = {}
        for v in set(valid_df[var]):
            v_df = valid_df.loc[valid_df[var] == v]
            dict_freq[v] = 1.0*v_df.shape[0]/df.shape[0]
            dict_bad_rate[v] = sum(v_df[target_var]*1.0)/v_df[target_var].shape[0]
        if df.loc[df[var] != df[var]][target_var].shape[0] > 0:
            dict_freq['missValue'] = 1.0-valid_df.shape[0]/df.shape[0]
            dict_bad_rate['missValue'] = \
                sum(df.loc[df[var] != df[var]][target_var]) / df.loc[df[var] != df[var]][target_var].shape[0]
        desc_state = pd.DataFrame({'percent': dict_freq, 'bad rate': dict_bad_rate})
        # 画图
        fig = pyplot.figure()
        ax0 = fig.add_subplot(111)
        ax1 = ax0.twinx()
        pyplot.title('The percentage and bad rate for '+var+'\n valid rec_perc ='+rec_perc_fmt)
        desc_state.percent.plot(kind='bar', color='blue', ax=ax0, width=0.2, position=1)
        desc_state['bad rate'].plot(kind='line', color='red', ax=ax1)
        ax0.set_ylabel('percent')
        ax1.set_ylabel('bad rate')
        fig_save_path = path+'\\'+str(var)+'.png'
        pyplot.savefig(fig_save_path)
        pyplot.close(1)


def makeup_miss_for_num(df, var_list, method):
    """
    按照指定的方法对数据集的数字类型数据进行填充
    :param df:数据集
    :param var_list:数字型变量名称list
    :param method:填充方法 'mean' 'RANDOM' 'PERC50'
    :return:已经填充的数据集
    """
    # df=dataAll
    # var_list=var_list
    # method='MEAN'
    # col= 'age1'

    df_makeup_miss = df.copy()
    if method.upper() not in ['MEAN', 'RANDOM', 'PERC50']:
        print('Please specify the correct treatment method for missing continuous variable:Mean or Random or PERC50!')
        return df_makeup_miss

    for col in set(var_list):
        valid_df = df_makeup_miss.loc[~np.isnan(df[col])][[col]]
        if valid_df.shape[0] == df_makeup_miss.shape[0]:
            continue
        # 得到列索引
        index_col = list(df_makeup_miss.columns).index(col)

        desc_state = valid_df[col].describe()
        value_mu = desc_state['mean']
        # std = desc_state['std']
        # max = desc_state['max']
        value_perc50 = desc_state['50%']
        # 盖帽
        # if max > mu+3*std:
        #     for i in list(valid_df.index):
        #         if valid_df.loc[i][col] > max:
        #             valid_df.loc[i][col] = mu+3*std
        #     # 重新计算mean
        #     mu = valid_df[col].describe()['mean']
        # makeup missing
        for i in range(df_makeup_miss.shape[0]):
            # if np.isnan(df.loc[i][col]):
            if np.isnan(df_makeup_miss[col][i]):
                if method.upper() == 'PERC50':
                    df_makeup_miss.iloc[i, index_col] = value_perc50
                elif method.upper() == 'MEAN':
                    df_makeup_miss.iloc[i, index_col] = value_mu
                elif method.upper() == 'RANDOM':
                    df_makeup_miss.iloc[i, index_col] = random.sample(valid_df[col], 1)[0]

    return df_makeup_miss


def makeup_miss_for_str(df, str_var_list, method):
    """
    按照指定的方法对数据集的字符串类型数据进行填充
    :param df: 数据集
    :param str_var_list: 字符类型变量名称list
    :param method: 填充方法 'MODE' 'RANDOM'
    :return: 已经填充的数据集
    """
    df_makeup_miss = df.copy()

    for var in str_var_list:
        valid_df = df_makeup_miss.loc[df_makeup_miss[var] == df_makeup_miss[var]][[var]]

        if valid_df.shape[0] == df_makeup_miss.shape[0]:
            continue

        index_var = list(df_makeup_miss.columns).index(var)

        if method.upper() == "MODE":
            dict_var_freq = {}
            num_recd = valid_df.shape[0]
            for v in set(valid_df[var]):
                dict_var_freq[v] = valid_df.loc[valid_df[var] == v].shape[0]*1.0/num_recd
            mode_val = max(dict_var_freq.items(), key=lambda x: x[1])[0]
            df_makeup_miss[var].fillna(mode_val, inplace=True)
        elif method.upper() == "RANDOM":
            list_dict = list(set(valid_df[var]))
            for i in range(df_makeup_miss.shape[0]):
                if df_makeup_miss.loc[i][var] != df_makeup_miss.loc[i][var]:
                    df_makeup_miss.iloc[i, index_var] = random.choice(list_dict)

    return df_makeup_miss


def encoder(df, col, target):
    """
    对类别变量进行密度编码 包括Nan
    :param df: 数据集
    :param col: 要分析的类别型变量的变量名
    :param target: 响应变量名
    :return: 返回每个类别对应的响应率
    """
    # df = data_all
    # col = 'marital'
    # target = col_target

    dict_encoder = {}
    for v in set(df[col]):
        if v == v:
            sub_df = df[df[col] == v]
        else:
            xlist = list(df[col])
            nan_ind = [i for i in range(len(xlist)) if xlist[i] != xlist[i]]
            sub_df = df.loc[nan_ind]
        dict_encoder[v] = sum(sub_df[target]) * 1.0 / sub_df.shape[0]
    new_col = [dict_encoder[i] for i in df[col]]
    return new_col


if __name__ == '__main__':
    # ##########################################################
    # #################原始数据处理              #################
    # ##########################################################
    # 根目录
    path_root = os.getcwd()
    # 子路径
    path_explore_result = path_root+'\\ExploreResult'
    # ID的字段名
    col_id = 'CUST_ID'
    # 目标字段名
    col_target = 'CHURN_CUST_IND'

    # 合并数据
    data_bank = pd.read_csv(path_root + '\\bankChurn.csv')
    data_external = pd.read_csv(path_root + '\\ExternalData.csv')
    data_all = pd.merge(data_bank, data_external, on=col_id)
    # #########################################################
    # ###              数据探索                     #############
    # #########################################################
    # 得到类别型和数字型变量名列表并保存
    string_var_list, number_var_list = get_list_for_number_str_col(df=data_all, col_id=col_id, col_target=col_target)
    list2txt(path_explore_result, 'string_var_list.txt', string_var_list)
    list2txt(path_explore_result, 'number_var_list.txt', number_var_list)
    # data_all[string_var_list]
    # data_all[number_var_list]
    # todo 调用小程序手动调整
    # todo 如果重新跑数据 或者调整字段则 用txt2list()重新加载即可

    # 分别进行数字型变量和字符串变量的探索
    num_var_perf(df=data_all, var_list=number_var_list, target_var=col_target, path=path_explore_result)
    str_var_pref(df=data_all, var_list=string_var_list, target_var=col_target, path=path_explore_result)
    # 选择15个数字变量 看相关性
    corr_cols = random.sample(number_var_list, 15)
    sample_df = data_all[corr_cols]
    scatter_matrix(sample_df, alpha=0.2, figsize=(14, 8), diagonal='kde')
    plt.show()

    # 缺失值填充
    allNoMissDate = makeup_miss_for_num(df=data_all, var_list=number_var_list, method='perc50')
    num_var_perf(df=allNoMissDate, var_list=number_var_list, target_var=col_target, path=path_explore_result+'\\1')
    allNoMissDate = makeup_miss_for_str(df=allNoMissDate, str_var_list=string_var_list, method='random')
    str_var_pref(df=allNoMissDate, var_list=string_var_list, target_var=col_target, path=path_explore_result+'\\1')
