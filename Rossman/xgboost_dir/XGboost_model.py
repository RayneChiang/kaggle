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


def find_low_high(feature, data):
    # find store specific Q1 - 3*IQ and Q3 + 3*IQ
    IQ = data.groupby('Store')[feature].quantile(0.75) - data.groupby('Store')[feature].quantile(0.25)
    Q1 = data.groupby('Store')[feature].quantile(0.25)
    Q3 = data.groupby('Store')[feature].quantile(0.75)
    low = Q1 - 3 * IQ
    high = Q3 + 3 * IQ
    low = low.to_frame()
    low = low.reset_index()
    low = low.rename(columns={feature: "low"})
    high = high.to_frame()
    high = high.reset_index()
    high = high.rename(columns={feature: "high"})
    return {'low': low, 'high': high}


def find_outlier_index(feature, data):
    main_data = data[['Store', feature]]
    low = find_low_high(feature, data)["low"]
    high = find_low_high(feature, data)["high"]

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


def get_store_sales_statistics(df, df2):
    mean = df.groupby('Store')['Sales'].mean()
    std = df.groupby('Store')['Sales'].std()
    mean_dataframe = pd.DataFrame(mean).reset_index()
    std_dataframe = pd.DataFrame(std).reset_index()
    df2 = pd.merge(df2, mean_dataframe, on='Store', how='left').rename(columns={"Sales": "SalesMean"})
    df2 = pd.merge(df2, std_dataframe, on='Store', how='left').rename(columns={"Sales": "SalesStd"})
    return df2


def get_sales_level_groups(df2):
    Q1 = df2.SalesMean.quantile(0.25)
    Q2 = df2.SalesMean.quantile(0.50)
    Q3 = df2.SalesMean.quantile(0.75)
    df2['StoreGroup1'] = (df2.SalesMean < Q1).astype(int)
    df2['StoreGroup2'] = ((df2.SalesMean >= Q1) & (df2.SalesMean < Q2)).astype(int)
    df2['StoreGroup3'] = ((df2.SalesMean >= Q2) & (df2.SalesMean < Q3)).astype(int)
    df2['StoreGroup4'] = (df2.SalesMean >= Q3).astype(int)
    df2['StoreGroup'] = df2['StoreGroup1'] + 2 * df2['StoreGroup2'] + 3 * df2['StoreGroup3'] + 4 * df2['StoreGroup4']
    df2.drop(['StoreGroup1', 'StoreGroup2', 'StoreGroup3', 'StoreGroup4'], axis=1, inplace=True)
    return df2


def build_features(features, data):
    data.loc[data.Open.isnull(), 'Open'] = 1
    features.extend(['Store', 'CompetitionDistance', 'Promo', 'Promo2', 'SchoolHoliday'])
    features.extend(['StoreType', 'Assortment', 'StateHoliday'])
    features.extend(['SalesPerDay', 'CustomersPerDay', 'SalesPerCustomersPerDay', 'StoreGroup'])

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


def create_new_store_feature(data, train):
    # create store group feature
    data = get_store_sales_statistics(train_data, data)
    data = get_sales_level_groups(data)
    # create some new feature on store related to sales
    store_data_sales = train.groupby([data['Store']])['Sales'].sum()
    store_data_customers = train.groupby([data['Store']])['Customers'].sum()
    store_data_open = train.groupby([data['Store']])['Open'].count()

    store_data_sales_per_day = store_data_sales / store_data_open
    store_data_customers_per_day = store_data_customers / store_data_open
    store_data_sales_per_customer_per_day = store_data_sales_per_day / store_data_customers_per_day

    data = pd.merge(data, store_data_sales_per_day.reset_index(name='SalesPerDay'), how='left', on=['Store'])
    data = pd.merge(data, store_data_customers_per_day.reset_index(name='CustomersPerDay'), how='left',
                    on=['Store'])
    data = pd.merge(data, store_data_sales_per_customer_per_day.reset_index(name='SalesPerCustomersPerDay'),
                    how='left', on=['Store'])

    return data


if __name__ == '__main__':
    path_dir = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
    test_data = pd.read_csv(path_dir + "/input/test.csv", parse_dates=[3])
    train_data = pd.read_csv(path_dir + "/input/train.csv", parse_dates=[2])
    store_data = pd.read_csv(path_dir + "/input/store.csv")

    # Drop duplicates
    train_data = train_data.drop_duplicates()
    store_data = store_data.drop_duplicates()
    # drop sales == 0 observations
    train_data = train_data[train_data.Sales != 0]
    # competition = store_data[store_data['CompetitionDistance'].isna()]
    df_open = test_data[test_data['Open'].isna()]
    test_data.fillna(1, inplace=True)  # assume all store open in test data
    store_data.fillna(0, inplace=True)


    train_data = train_data.reset_index()
    train_data.drop(find_outlier_index("Sales", train_data), inplace=True, axis=0)


    store_data = create_new_store_feature(store_data, train_data)

    # Join store_data
    train_data = pd.merge(train_data, store_data, on='Store')
    test_data = pd.merge(test_data, store_data, on='Store')

    # argument feature
    features = []
    train_data, features = build_features(features, train_data)
    test_data, _ = build_features([], test_data)

    # train XBGoost model
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
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, \
                    early_stopping_rounds=100, feval=rmspe_xg, verbose_eval=True)

    print("Validating")
    yhat = gbm.predict(xgb.DMatrix(X_valid[features]))
    error = rmspe(X_valid.Sales.values, np.expm1(yhat))
    print('RMSPE: {:.6f}'.format(error))

    # use the trained model to predict
    dtest = xgb.DMatrix(test_data[features])
    test_probs = gbm.predict(dtest)
    result = pd.DataFrame({"Id": test_data["Id"], 'Sales': np.expm1(test_probs)})

    result[result < 0] = 0
    result = result.sort_values(by='Id', ascending=True)

    # save final submission
    result.to_csv("xgboost_submission.csv", index=False)

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
