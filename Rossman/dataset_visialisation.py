import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
pd.set_option('display.max_columns', None)

SEED = 42

def display_missing(df):
    for col in df.columns.tolist():
        print('{} column missing values: {}'.format(col, df[col].isnull().sum()))
    print('\n')


if __name__ == '__main__':
    test_data = pd.read_csv("test.csv")
    train_data = pd.read_csv("train.csv")
    sample_data = pd.read_csv("sample_submission.csv")
    store_data = pd.read_csv("store.csv")

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

