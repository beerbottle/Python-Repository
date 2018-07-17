# Index
# ----------------------------------------
# makeup_miss_for_num     按照指定的方法对数据集的数字类型数据进行填充
# makeup_miss_for_str     按照指定的方法对数据集的字符串类型数据进行填充
# density_encoder        对类别变量进行密度编码 包括Nan

def makeup_miss_for_num(p_df, p_var_list, p_method):
    """
    按照指定的方法对数据集的数字类型数据进行填充
    :param p_df:数据集
    :param p_var_list:数字型变量名称list
    :param p_method:填充方法 'MEAN' 'RANDOM' 'PERC50'
    :return:已经填充的数据集
    """
    import random
    import numpy as np
    # df=dataAll
    # var_list=var_list
    # method='MEAN'
    # col= 'age1'

    df_makeup_miss = p_df.copy()
    if p_method.upper() not in ['MEAN', 'RANDOM', 'PERC50', "3STDCAP"]:
        print('Please specify the correct treatment method for missing continuous variable:Mean or Random or ' +
              'PERC50 or 3STDCAP!')
        return df_makeup_miss

    for col in set(p_var_list):
        valid_df = df_makeup_miss.loc[~np.isnan(p_df[col])][[col]]
        if valid_df.shape[0] == df_makeup_miss.shape[0]:
            continue
        # 得到列索引
        index_col = list(df_makeup_miss.columns).index(col)

        desc_state = valid_df[col].describe()
        value_mu = desc_state['mean']
        # max = desc_state['max']
        value_perc50 = desc_state['50%']
        # std = desc_state['std']
        # value_3stdCap = value_mu + 3*std
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
                if p_method.upper() == 'PERC50':
                    df_makeup_miss.iloc[i, index_col] = value_perc50
                elif p_method.upper() == 'MEAN':
                    df_makeup_miss.iloc[i, index_col] = value_mu
                elif p_method.upper() == 'RANDOM':
                    df_makeup_miss.iloc[i, index_col] = random.sample(valid_df[col], 1)[0]

    print("function makeup_miss_for_num finished!...................")
    return df_makeup_miss


def makeup_miss_for_str(p_df, p_str_var_list, p_method):
    """
    按照指定的方法对数据集的字符串类型数据进行填充
    :param p_df: 数据集
    :param p_str_var_list: 字符类型变量名称list
    :param p_method: 填充方法 'MODE' 'RANDOM'
    :return: 已经填充的数据集
    """
    import random

    df_makeup_miss = p_df.copy()

    if p_method.upper() not in ['MODE', 'RANDOM']:
        print('Please specify the correct treatment method for missing continuous variable:MODE or Random!')
        return df_makeup_miss

    for var in p_str_var_list:
        valid_df = df_makeup_miss.loc[df_makeup_miss[var] == df_makeup_miss[var]][[var]]

        if valid_df.shape[0] == df_makeup_miss.shape[0]:
            continue

        index_var = list(df_makeup_miss.columns).index(var)

        if p_method.upper() == "MODE":
            dict_var_freq = {}
            num_recd = valid_df.shape[0]
            for v in set(valid_df[var]):
                dict_var_freq[v] = valid_df.loc[valid_df[var] == v].shape[0]*1.0/num_recd
            mode_val = max(dict_var_freq.items(), key=lambda x: x[1])[0]
            df_makeup_miss[var].fillna(mode_val, inplace=True)
        elif p_method.upper() == "RANDOM":
            list_dict = list(set(valid_df[var]))
            for i in range(df_makeup_miss.shape[0]):
                if df_makeup_miss.loc[i][var] != df_makeup_miss.loc[i][var]:
                    df_makeup_miss.iloc[i, index_var] = random.choice(list_dict)

    print("function makeup_miss_for_str finished!...................")
    return df_makeup_miss


def density_encoder(p_df, p_col, p_target):
    """
    对类别变量进行密度编码 包括Nan
    :param p_df: 数据集
    :param p_col: 要分析的类别型变量的变量名
    :param p_target: 响应变量名
    :return: 返回每个类别对应的响应率
    """
    # df = data_all
    # col = 'marital'
    # target = col_target

    dict_encoder = {}
    for v in set(p_df[p_col]):
        if v == v:
            sub_df = p_df[p_df[p_col] == v]
        else:
            xlist = list(p_df[p_col])
            nan_ind = [i for i in range(len(xlist)) if xlist[i] != xlist[i]]
            sub_df = p_df.loc[nan_ind]
        dict_encoder[v] = sum(sub_df[p_target]) * 1.0 / sub_df.shape[0]
    new_col = [dict_encoder[i] for i in p_df[p_col]]
    print("function density_encoder finished!...................")
    return new_col

