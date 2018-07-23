#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import random
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import chinaPnr.utility.explore as u_explore
import chinaPnr.utility.modify as u_modify
import chinaPnr.utility.others as u_others
import chinaPnr.utility.sample as u_sample
import chinaPnr.utility.model as u_model
import chinaPnr.utility.assess as u_assess
# import io
# import sys
# import numbers
# import numpy as np
# from matplotlib import pyplot


if __name__ == '__main__':
    # ##########################################################
    # #################原始数据处理              #################
    # ##########################################################
    # 根目录
    path_root = os.getcwd()
    # 路径
    path_explore_result = path_root+'\\Result\\Explore'
    u_others.create_path(path_explore_result)
    # ID的字段名
    col_id = 'CUST_ID'
    # 目标字段名
    col_target = 'CHURN_CUST_IND'

    # 合并数据
    data_bank = pd.read_csv(path_root + '\\bankChurn.csv')
    data_external = pd.read_csv(path_root + '\\ExternalData.csv')
    data_all = pd.merge(data_bank, data_external, on=col_id)
    data_all.head(5)
    # #########################################################
    # ###              数据探索                     #############
    # #########################################################
    # 得到类别型和数字型变量名列表并保存
    string_var_list, number_var_list = u_explore.get_list_for_number_str_col(p_df=data_all, p_col_id=col_id,
                                                                             p_col_target=col_target)
    u_others.list2txt(path_explore_result, "var_string_list.txt", string_var_list)
    u_others.list2txt(path_explore_result, "var_number_list.txt", number_var_list)
    # data_all[string_var_list]
    # data_all[number_var_list]
    # todo 调用小程序手动调整
    # todo 如果重新跑数据 或者调整字段则 用txt2list()重新加载即可
    # string_var_list = txt2list(path_explore_result+"\\var_string_list.txt")
    # number_var_list = txt2list(path_explore_result+"\\var_number_list.txt")

    # 分别进行数字型变量和字符串变量的探索
    u_explore.num_var_perf(p_df=data_all, p_var_list=number_var_list, p_target_var=col_target,
                           p_path=path_explore_result)
    u_explore.str_var_pref(p_df=data_all, p_var_list=string_var_list, p_target_var=col_target,
                           p_path=path_explore_result)

    # 选择15个数字变量 看相关性
    # corr_cols = random.sample(number_var_list, 15)
    # sample_df = data_all[corr_cols]
    # scatter_matrix(sample_df, alpha=0.2, figsize=(14, 8), diagonal='kde')
    # plt.show()

    # 缺失值填充
    u_modify.makeup_miss_for_num(p_df=data_all, p_var_list=number_var_list, p_method="MEAN")
    u_modify.makeup_miss_for_str(p_df=data_all, p_str_var_list=string_var_list, p_method="MODE")

    # 浓度编码
    # u_modify.density_encoder(data_all, string_var_list, col_target)

    # 卡方分箱
    u_model.chi2_bin(data_all, number_var_list, col_target)

