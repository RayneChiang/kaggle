import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import missingno
import os
from explore_data_analysis import eda

# preliminary data processing
path_dir = os.path.dirname(__file__)
# test_data = pd.read_csv(path_dir + "/input/test.csv")
df = pd.read_csv(path_dir + "/input/train.csv")
# store_data = pd.read_csv(path_dir + "/input/store.csv")

df.info()
df.head()

# set identifier "Store" as string
df['Store'] = df['Store'].astype('str')
# set categorical data
df['DayOfWeek'] = df['DayOfWeek'].astype('category')
df['Open'] = df['Open'].astype('category')
df['Promo'] = df['Promo'].astype('category')
df['StateHoliday'] = df['StateHoliday'].astype(str).str.strip().astype('category')
df['SchoolHoliday'] = df['SchoolHoliday'].astype('category')
# set datetime data
df['Date'] = pd.to_datetime(df['Date'])


eda(df)