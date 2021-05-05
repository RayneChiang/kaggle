import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import operator
import matplotlib
import os
from sklearn.model_selection import GridSearchCV

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


def find_low_high(feature):
    # find store specific Q1 - 3*IQ and Q3 + 3*IQ
    IQ = df.groupby('Store')[feature].quantile(0.75)-df.groupby('Store')[feature].quantile(0.25)
    Q1 = df.groupby('Store')[feature].quantile(0.25)
    Q3 = df.groupby('Store')[feature].quantile(0.75)
    low = Q1 - 3*IQ
    high = Q3 + 3*IQ
    low = low.to_frame()
    low = low.reset_index()
    low = low.rename(columns={feature: "low"})
    high = high.to_frame()
    high = high.reset_index()
    high = high.rename(columns={feature: "high"})
    return {'low':low, 'high':high}


def find_outlier_index(feature):
    main_data = df[['Store', feature]]
    low = find_low_high(feature)["low"]
    high = find_low_high(feature)["high"]

    new_low = pd.merge(main_data, low, on='Store', how='left')
    new_low['outlier_low'] = (new_low[feature] < new_low['low'])
    index_low = new_low[new_low['outlier_low'] == True].index
    index_low = list(index_low)

    new_high = pd.merge(main_data, high, on='Store', how='left')
    new_high['outlier_high'] = new_high[feature] > new_high['high']
    index_high = new_high[new_high['outlier_high'] == True].index
    index_high = list(index_high)

    index_low.extend(index_high)
    index = list(set(index_low))
    return index


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
    data['WeekOfYear'] = data.Date.dt.week

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

    # Drop duplicates
    train_data= train_data.drop_duplicates()
    store_data = store_data.drop_duplicates()
    # drop sales == 0 observations
    train_data = train_data[train_data.Sales != 0]
    # competition = store_data[store_data['CompetitionDistance'].isna()]
    df_open = test_data[test_data['Open'].isna()]
    test_data.fillna(1, inplace=True)  # assume all store open in test data
    store_data.fillna(0, inplace=True)

    train_data = train_data.reset_index()
    train_data.drop(find_outlier_index("Sales"), inplace=True, axis=0)

    # Join store_data
    train_data = pd.merge(train_data, store_data, on='Store')
    test_data = pd.merge(test_data, store_data, on='Store')

    # argument feature
    features = []
    train_data, features = build_features(features, train_data)
    test_data, _ = build_features([], test_data)

    # train XBGoost model
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

    # split dataset : year before 2015 as training set, after as validation set
    X_train = train_data[train_data['Year'] < 2015]
    X_valid = train_data[train_data['Year'] >= 2015]
    # So log1p produces only positive values and removes the 'danger' of large negative numbers.
    # This generally insures a more homogeneous distribution when a dataset contains numbers close to zero.
    y_train = np.log1p(X_train.Sales)
    y_valid = np.log1p(X_valid.Sales)
    dtrain = xgb.DMatrix(X_train[features], y_train)
    dvalid = xgb.DMatrix(X_valid[features], y_valid)

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    params_grid = {'max_depth': (3, 5, 10),
                   'colsample_bytree': (0.5, 0.7, 0.8),
                   'learning_rate': (0.1, 0.3),
                   'subsample': (0.7, 0.9),

                   # 'early_stopping_rounds': [100],
                   # "objective": ["reg:linear"],
                   # "booster": ["gbtree"],
                   # "eta": [0.3],
                   # 'evals' : [watchlist],
                   #  'feval' :[rmspe_xg],
                   # 'verbose_eval': [True]
                   }

    gbm = xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bynode=1, colsample_bytree=1, gamma=0,
       importance_type='gain', learning_rate=0.1, max_delta_step=0,
       max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
       n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=None, subsample=1, verbosity=1)
    xgb_grid = GridSearchCV(gbm, params_grid, n_jobs=5)
    xgb_grid.fit(X_train[features], y_train)

    print(xgb_grid.cv_results_)
    print("Validating")
    yhat = xgb_grid.predict(X_valid[features])
    error = rmspe(X_valid.Sales.values, np.expm1(yhat))
    print('RMSPE: {:.6f}'.format(error))

    # use the trained model to predict
    dtest = test_data[features]
    test_probs = xgb_grid.predict(dtest)
    result = pd.DataFrame({"Id": test_data["Id"], 'Sales': np.expm1(test_probs)})

    result[result < 0] = 0
    result = result.sort_values(by='Id', ascending=True)

    # save final submission
    result.to_csv("xgboost_submission.csv", index=False)

    # XGB feature importances Based on https://www.kaggle.com/mmueller/liberty-mutual-group-property-inspection
    # -prediction/xgb-feature-importance-python/code

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

    xgb.plot_tree(xgb_grid.best_estimator_)
    plt.rcParams['figure.figsize'] = [20, 10]
    plt.savefig('decision tree')