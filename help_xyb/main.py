import os
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import tree
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import operator
from sklearn.model_selection import train_test_split
from xgboost import plot_tree


def get_missing(df):
    for col in df.columns.tolist():
        df[col].fillna(round(df[col].mean()), inplace=True)
        print('{} column missing values: {}'.format(col, df[col].isnull().sum()))
    print('\n')

def get_NaNout(df):
    for col in df.columns.tolist():
        index_list = df[col][df[col] == '”NaN”'].index
        for i in index_list:
            df.loc[i, col] = round(df[col].mean())

def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()


if __name__ == '__main__':
    df = pd.read_csv('data_A3a_dropna.csv')
    columns = ['A3a', 'A2', 'A1', 'D1_a', 'D1_b', 'D1_c', 'D1_d', 'D1_e', 'D1_f',
               'A5_a', 'A5_b', 'A5_c', 'A5_d', 'A5_e', 'A5_f', 'A5_g',
               'A4_a', 'A4_b', 'A4_c', 'A4_d', 'A4_e', 'B5', 'B4', 'E14', 'p_age_group_sdc']
    # columns =
    df_select = df[df.columns[3:-7]]
    get_missing(df_select)
    # get_NaNout(df_select)
    y = df['Mental_Goal']
    X = df_select
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    #
    # xg_reg = xgb.XGBRegressor(objective='reg:linear', colsample_bytree=0.7, learning_rate=0.1,
    #                           max_depth=10, alpha=10, n_estimators=100)
    # xg_reg.fit(X_train, y_train)
    #
    # preds = xg_reg.predict(X_test)
    #
    # rmse = np.sqrt(mean_squared_error(y_test, preds))
    # print("RMSE: %f" % (rmse))


    # xgb.plot_tree(xg_reg, num_trees=0)
    # plt.rcParams['figure.figsize'] = [50, 10]
    # plt.show()



    features = df.columns[3:-7]
    params = {"objective": "reg:linear",
              "booster": "gbtree",
              "eta": 0.3,
              "max_depth": 10,
              "subsample": 0.9,
              "colsample_bytree": 0.7,
              "silent": 1,
              "seed": 1301
              }
    num_boost_round = 300

    dtrain = xgb.DMatrix(X, y)
    gbm = xgb.train(params, dtrain, num_boost_round)

    create_feature_map(features)
    importance = gbm.get_score(importance_type='weight')
    importance = sorted(importance.items(), key=operator.itemgetter(1))

    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()

    featp = df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
    plt.title('XGBoost Feature Importance')
    plt.xlabel('relative importance')
    fig_featp = featp.get_figure()
    fig_featp.savefig('feature_importance_xgb.png', bbox_inches='tight', pad_inches=1)

    # keys = list(importance.keys())
    # values = list(importance.values())
    #
    # data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by="score", ascending=True)
    # data.plot(kind='barh')
    # plt.show()