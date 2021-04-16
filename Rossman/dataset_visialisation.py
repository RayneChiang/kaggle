import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import os
from IPython.display import display
pd.set_option('display.max_columns', None)

SEED = 42

def display_missing(df):
    for col in df.columns.tolist():
        print('{} column missing values: {}'.format(col, df[col].isnull().sum()))
    print('\n')


if __name__ == '__main__':
    path_dir = os.path.dirname(__file__)
    test_data = pd.read_csv(path_dir + "/input/test.csv")
    train_data = pd.read_csv(path_dir + "/input/train.csv")
    store_data = pd.read_csv(path_dir + "/input/store.csv")
    print(test_data.describe())
    print(train_data.describe())
    print(store_data.describe())
    sample_data = pd.read_csv(path_dir + "/input/sample_submission.csv")
    # rename dataset
    train_data.name = 'Training Set'
    test_data.name = 'Test Set'
    store_data.name = 'Store Set'

    dfs = [train_data, test_data, store_data]
    for df in dfs:
        print('{}'.format(df.name))
        display_missing(df)

    display(train_data.head(3))
    print('\n')
    display(test_data.head(3))
    print('\n')
    display(store_data.head(3))
    print('\n')
    display(sample_data.head(3))

    store_1 = train_data.loc[(train_data["Store"] == 1) & (train_data['Sales'] > 0), ['Date', "Sales"]]
    store_2 = train_data.loc[(train_data["Store"] == 2) & (train_data['Sales'] > 0), ['Date', "Sales"]]
    # f = plt.figure(figsize=(18, 10))
    # ax1 = f.add_subplot(211)
    # ax1.plot(store_1['Date'], store_1['Sales'], '-')
    # ax1.set_xlabel('Time')
    # ax1.set_ylabel('Sales')
    # ax1.xaxis.set_major_locator(ticker.AutoLocator())
    # ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    # # ax1.text(0.0, 0.1, "AutoLocator()", fontsize=14, transform=ax1.transAxes)
    #
    # ax1.set_title('Store 1 Sales Distribution')
    #
    # ax2 = f.add_subplot(212)
    # ax2.plot(store_2['Date'], store_2['Sales'], '-')
    # ax2.set_xlabel('Time')
    # ax2.set_ylabel('Sales')
    # ax2.xaxis.set_major_locator(ticker.AutoLocator())
    # ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    # ax2.set_title('Store 2 Sales Distribution')
    # plt.savefig('Store Sales Distribution')

    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set()
    # check distribution of sales in train set
    # fig = plt.figure(figsize=(12, 5))
    # ax1 = fig.add_subplot(121)
    # ax2 = fig.add_subplot(122)
    # sales_view = train_data[train_data['Sales'] > 0]['Sales']
    # g1 = sns.histplot(sales_view, label='skewness:{:.2f}'.format(train_data['Sales'].skew()), ax=ax1)
    # g1.legend()
    # g1.set(xlabel='Sales', ylabel='Density', title='Sales Distribution')
    # g2 = sns.histplot(np.log1p(sales_view), label='skewness:{:.2f}'.format(np.log1p(train_data['Sales']).skew()), ax=ax2)
    # g2.legend()
    # g2.set(xlabel='log(Sales+1)', ylabel='Density', title='log(Sales+1) Distribution')
    # plt.savefig('sales Distribution')
    # print('sales saved')
    #
    # fig = plt.figure(figsize=(12, 5))
    # ax1 = fig.add_subplot(121)
    # ax2 = fig.add_subplot(122)
    # sales_view = train_data[train_data['Customers'] > 0]['Customers']
    # g1 = sns.histplot(sales_view, label='skewness:{:.2f}'.format(train_data['Customers'].skew()), ax=ax1)
    # g1.legend()
    # g1.set(xlabel='Customer', ylabel='Density', title='Customer Distribution')
    # g2 = sns.histplot(np.log1p(sales_view), label='skewness:{:.2f}'.format(np.log1p(train_data['Sales']).skew()), ax=ax2)
    # g2.legend()
    # g2.set(xlabel='log(Customers+1)', ylabel='Density', title='log(Customers+1) Distribution')
    # plt.savefig('customers Distribution')
    # print('customers saved')


    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    sales_view = store_data[store_data['Customers'] > 0]['Customers']
    g1 = sns.histplot(sales_view, label='skewness:{:.2f}'.format(train_data['Customers'].skew()), ax=ax1)
    g1.legend()
    g1.set(xlabel='Customer', ylabel='Density', title='Customer Distribution')
    g2 = sns.histplot(np.log1p(sales_view), label='skewness:{:.2f}'.format(np.log1p(train_data['Sales']).skew()), ax=ax2)
    g2.legend()
    g2.set(xlabel='log(Customers+1)', ylabel='Density', title='log(Customers+1) Distribution')
    plt.savefig('customers Distribution')
    print('customers saved')