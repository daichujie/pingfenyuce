
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import preprocessing # 机器学习库中数据预处理的模块，可以将字符串，文本转换成对应的数字
from matplotlib import style

style.use('ggplot')     # 设置图片显示的主题样式
# 解决matplotlib显示中文问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题



# 1. 探索性数据分析的流程
def inspect_data(df_data):
    #　查看数据的前５行
    print(df_data.head())
    print("++++++++++++++++++")
    #　查看数据的后5行
    print(df_data.tail())
    print("++++++++++++++++++")
    #  显示数据的基本信息
    print(df_data.info())
    print("++++++++++++++++++")
    # 显示数据的统计信息
    print(df_data.describe())

# 2. 数据的分析及画图（pandas）
def analysis_data(df_data):
    use_cols = ['movie_1','agreeableness','emotional_stability','predicted_rating_1',
                'is_personalized','openness','enjoy_watching']  #修改
    use_data = df_data[use_cols]
    print("\n数据分析总览，查看使用列数据的前5行")
    print(use_data.head())

    print("+++++++++++++++++++++++++++++++++++++++")
    print('\n各喜欢观看的数据量')
    print(use_data['enjoy_watching'].value_counts()) #　统计不同喜爱观看的总量

    # 图1
    # 2. 按照个性化统计喜爱总量
    loan_amout_group_by_df = use_data.groupby(['is_personalized'])['enjoy_watching'].sum()

    # 可视化，喜爱总量vs个性化
    loan_amout_group_by_df.plot()
    plt.xlabel('个性化')
    plt.ylabel('喜爱总量')
    plt.title('个性化 vs 喜爱总量')
    plt.tight_layout()
    # plt.savefig('loan_amount_vs_month.png')
    plt.show()

    # 图2
    print("+++++++++++++++++++++++++++++++++")
    # 3. 按照opennness统计movie_1总量
    data_group_by_state = use_data.groupby(['openness'])['movie_1'].sum()

    # 3.1 可视化， openness和movie_1
    data_group_by_state.plot(kind='bar')
    plt.xlabel("openness")
    plt.ylabel("movie_1")
    plt.title("openness vs movie_1")
    plt.tight_layout()
    plt.show()

    # 图3
    # 4 emotional_stability','agreeableness'和predicted_rating_1的关系

    data_group_by_agreeableness = use_data.groupby(['emotional_stability','agreeableness'])['predicted_rating_1'].mean() # 看平均值
    data_group_by_agreeableness_df = pd.DataFrame(data_group_by_agreeableness).reset_index()

    data_group_by_agreeableness_df.plot()
    plt.xlabel("emotional_stability','agreeableness'")
    plt.ylabel("predicted_rating_1")
    plt.title("emotional_stability','agreeableness' vs predicted_rating_1")
    plt.tight_layout()
    plt.show()

    print("\nemotional_stability','agreeableness'和predicted_rating_1的关系预览：")
    print(data_group_by_agreeableness_df.head())  
    print(data_group_by_agreeableness_df.info())


     # 保存	
    df = pd.DataFrame(data_group_by_agreeableness_df)
    df.to_csv('pre.csv')


def main():
    raw_data = pd.read_csv("2018-personality-data.csv")
    #inspect_data(raw_data)
    analysis_data(raw_data)


if __name__ == '__main__':
    main()