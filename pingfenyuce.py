
import numpy as np
import pandas as pd

dtype = {'userId': np.int32, 'movieId': np.int32, 'rating': np.float32}
ratings = pd.read_csv('ratings.csv', dtype=dtype, usecols=range(3))

# 构建透视表 找到用户和电影之间的评分关系
ratings_matrix = ratings.pivot_table(values=['rating'], index=['userId'], columns=['movieId'])
ratings_matrix

# 计算用户-物品相似度（皮尔逊相关系数）
# 用户：
user_sim = ratings_matrix.T.corr()


def predict(uid, iid, ratings_matrix, user_sim):
    """
    预测给定用户对给定物品的评分
    """
    # 1.找出uid用户的相似用户
    sim_users = user_sim[uid].drop([uid]).dropna()
    # 相似用户筛选规则，正相关用户
    sim_users = sim_users.where(sim_users > 0).dropna()

    # 2.从uid用户近邻相似用户中筛选对iid有过评分的用户
    ids = set(ratings_matrix.loc[:, ('rating', iid)].dropna().index) & set(sim_users.index)
    finally_sim_users = sim_users.loc[list(ids)]

    # 3.结合uid用户与其近邻用户的相似度预测uid用户对iid物品的评分
    sum_up = 0  # 评分预测公式分子
    sum_down = 0  # 评分预测公式分母
    for sim_uid, sim_v in finally_sim_users.items():
        # 近邻用户的评分数据
        sim_user_rated_movies = ratings_matrix.loc[sim_uid].dropna()
        # 近邻用户对iid物品评分
        sim_user_rating_for_item = sim_user_rated_movies[iid]
        # 计算分子分母
        sum_up += sim_v * sim_user_rating_for_item
        sum_down += sim_v

    # 计算预测的评分
    pred_rating = sum_up / sum_down
    print(f"预测出用户{uid}对电影{iid}评分:{pred_rating}")
    return round(pred_rating, 2)


def predict_all(uid, ratings_matrix, user_sim):
    # 预测全部评分
    items_ids = ratings_matrix.columns
    for iid in items_ids:
        try:
            rating = predict(uid, iid, ratings_matrix, user_sim)
        except Exception as e:
            print(e)
        else:
            yield uid, iid, rating



# 预测用户1对所有物品的评分
for i in predict_all(1, ratings_matrix, user_sim):
    pass
    break

