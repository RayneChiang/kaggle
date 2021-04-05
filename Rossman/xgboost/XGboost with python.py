import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import operator
import matplotlib
import os

matplotlib.use('Agg')

import matplotlib.pyplot as plt


def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()


def rmspe(y, yhat):
    return np.sqrt(np.mean((yhat / y - 1) ** 2))


def rmspe_xg(yhat, y):
    y = np.expm1(y.get_label())
    yhat = np.expm1(yhat)
    return "rmspe", rmspe(y, yhat)


def build_features(features, data):
    data.loc[data.Open.isnull(), 'Open'] = 1
    features.extend(['Store', 'CompetitionDistance', 'Promo', 'Promo2', 'SchoolHoliday'])
    features.extend(['StoreType', 'Assortment', 'StateHoliday'])

    mappings = {'0': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4}
    data.StoreType.replace(mappings, inplace=True)
    data.Assortment.replace(mappings, inplace=True)
    data.StateHoliday.replace(mappings, inplace=True)

    features.extend(['DayOfWeek', 'Month', 'Day', 'Year', 'WeekOfYear'])
    data['Year'] = data.Date.dt.year
    data['Month'] = data.Date.dt.month
    data['Day'] = data.Date.dt.day
    data['DayOfWeek'] = data.Date.dt.dayofweek
    data['WeekOfYear'] = data.Date.dt.weekofyear

    # CompetionOpen en PromoOpen from https://www.kaggle.com/ananya77041/rossmann-store-sales/randomforestpython/code
    # Calculate time competition open time in months
    features.append('CompetitionOpen')
    data['CompetitionOpen'] = 12 * (data.Year - data.CompetitionOpenSinceYear) + \
                              (data.Month - data.CompetitionOpenSinceMonth)
    # Promo open time in months
    features.append('PromoOpen')
    data['PromoOpen'] = 12 * (data.Year - data.Promo2SinceYear) + \
                        (data.WeekOfYear - data.Promo2SinceWeek) / 4.0
    data['PromoOpen'] = data.PromoOpen.apply(lambda x: x if x > 0 else 0)
    data.loc[data.Promo2SinceYear == 0, 'PromoOpen'] = 0

    # Indicate that sales on that day are in promo interval
    features.append('IsPromoMonth')
    month2str = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                 7: 'Jul', 8: 'Aug', 9: 'Sept', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    data['monthStr'] = data.Month.map(month2str)
    data.loc[data.PromoInterval == 0, 'PromoInterval'] = ''
    data['IsPromoMonth'] = 0
    for interval in data.PromoInterval.unique():
        if interval != '':
            for month in interval.split(','):
                data.loc[(data.monthStr == month) & (data.PromoInterval == interval), 'IsPromoMonth'] = 1

    return data, features


if __name__ == '__main__':
    path_dir = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
    test_data = pd.read_csv(path_dir + "/input/test.csv", parse_dates=[3])
    train_data = pd.read_csv(path_dir + "/input/train.csv", parse_dates=[2])
    store_data = pd.read_csv(path_dir + "/input/store.csv")

    # store_data.astype(types)

    # clean na data
    print(train_data.isna().sum())
    print(test_data.isna().sum())
    print(store_data.isna().sum())

    # competition = store_data[store_data['CompetitionDistance'].isna()]
    print(train_data.Open.value_counts())
    print(test_data.Open.value_counts())
    df_open = test_data[test_data['Open'].isna()]
    test_data.fillna(1, inplace=True)  # assume all store open in test data
    store_data.fillna(0, inplace=True)
    print(store_data.dtypes)

    # types = {
    #     'CompetitionOpenSinceYear': np.dtype(int),
    #     'CompetitionOpenSinceMonth': np.dtype(int),
    #     'StateHoliday': np.dtype(str),
    #     'Promo2SinceWeek': np.dtype(int),
    #     'SchoolHoliday': np.dtype(float),
    #     'PromoInterval': np.dtype(str)}
    # store_data.astype(dtype=types)

    # Join store_data
    train_data = pd.merge(train_data, store_data, on='Store')
    test_data = pd.merge(test_data, store_data, on='Store')

    # argument feature
    features = []
    train_data, features = build_features(features, train_data)
    test_data, _ = build_features([], test_data)

    print('training data processed')

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

    print("Train a XGBoost model")
    # X_train, X_valid = train_test_split(train_data, test_size=0.2, random_state=10)
    X_train = train_data[train_data['Year'] < 2015]
    X_valid = train_data[train_data['Year'] >= 2015]
    # So log1p produces only positive values and removes the 'danger' of large negative numbers.
    # This generally insures a more homogeneous distribution when a dataset contains numbers close to zero.
    y_train = np.log1p(X_train.Sales)
    y_valid = np.log1p(X_valid.Sales)
    dtrain = xgb.DMatrix(X_train[features], y_train)
    dvalid = xgb.DMatrix(X_valid[features], y_valid)

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist,\
                    early_stopping_rounds=500, feval=rmspe_xg, verbose_eval=True)

    print("Validating")
    yhat = gbm.predict(xgb.DMatrix(X_valid[features]))
    error = rmspe(X_valid.Sales.values, np.expm1(yhat))
    print('RMSPE: {:.6f}'.format(error))

    print("Make predictions on the test set")
    dtest = xgb.DMatrix(test_data[features])
    test_probs = gbm.predict(dtest)
    # Make Submission
    result = pd.DataFrame({"Id": test_data["Id"], 'Sales': np.expm1(test_probs)})

    result[result < 0] = 0
    result = result.sort_values(by='Id', ascending=True)


    result.to_csv("xgboost_submission.csv", index=False)

    # XGB feature importances
    # Based on https://www.kaggle.com/mmueller/liberty-mutual-group-property-inspection-prediction/xgb-feature-importance-python/code

    create_feature_map(features)
    importance = gbm.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=operator.itemgetter(1))

    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()

    featp = df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
    plt.title('XGBoost Feature Importance')
    plt.xlabel('relative importance')
    fig_featp = featp.get_figure()
    fig_featp.savefig('feature_importance_xgb.png', bbox_inches='tight', pad_inches=1)

