'''
Try to enhance on the result from Xgboost
'''
import lightgbm as lgb
import os
import pandas as pd
import numpy as np
import operator
import matplotlib as plt
from XGboost_model import rmspe, rmspe_xg, build_features, create_feature_map

if __name__ == '__main__':
    path_dir = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
    test_data = pd.read_csv(path_dir + "/input/test.csv", parse_dates=[3])
    train_data = pd.read_csv(path_dir + "/input/train.csv", parse_dates=[2])
    store_data = pd.read_csv(path_dir + "/input/store.csv")

    df_open = test_data[test_data['Open'].isna()]
    test_data.fillna(1, inplace=True)  # assume all store open in test data
    store_data.fillna(0, inplace=True)

    # Join store_data
    train_data = pd.merge(train_data, store_data, on='Store')
    test_data = pd.merge(test_data, store_data, on='Store')

    features = []
    train_data, features = build_features(features, train_data)
    test_data, _ = build_features([], test_data)

    # train LightGBM model
    params = {
        "max_depth": 5,
        "learning_rate": 0.05,
        "num_leaves": 500,
        "n_estimators": 300,
        "seed": 1301
    }

    X_train = train_data[train_data['Year'] < 2015]
    X_valid = train_data[train_data['Year'] >= 2015]

    y_train = np.log1p(X_train.Sales)
    y_valid = np.log1p(X_valid.Sales)
    dtrain = lgb.Dataset(X_train[features], label=X_train.Sales)
    dvalid = lgb.Dataset(X_valid[features], label=X_valid.Sales)

    model_lgb = lgb.train(params, dtrain, valid_sets=[dvalid])

    # use the trained model to predict
    test_probs = model_lgb.predict(test_data[features])
    result = pd.DataFrame({"Id": test_data["Id"], 'Sales': test_probs})

    result[result < 0] = 0
    result = result.sort_values(by='Id', ascending=True)

    # save final submission
    result.to_csv("LGBM_submission.csv", index=False)
