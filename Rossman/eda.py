import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import missingno
import os
from explore_data_analysis import eda,time_series_plot

# preliminary data processing
path_dir = os.path.dirname(__file__)
test_data = pd.read_csv(path_dir + "/input/test.csv")
train_data = pd.read_csv(path_dir + "/input/train.csv")
store_data = pd.read_csv(path_dir + "/input/store.csv")

train_data = pd.merge(train_data, store_data, on='Store')
test_data = pd.merge(test_data, store_data, on='Store')

# set identifier "Store" as string
train_data['Store'] = train_data['Store'].astype('str')
# set categorical data
train_data['DayOfWeek'] = train_data['DayOfWeek'].astype('category')
train_data['Open'] = train_data['Open'].astype('category')
train_data['Promo'] = train_data['Promo'].astype('category')
train_data['StateHoliday'] = train_data['StateHoliday'].astype(str).str.strip().astype('category')
train_data['SchoolHoliday'] = train_data['SchoolHoliday'].astype('category')

# set datetime data
train_data['Date'] = pd.to_datetime(train_data['Date'])

# from pylab import rcParams
# import statsmodels.api as sm
#
# rcParams['figure.figsize'] = 11, 9
# decomposition = sm.tsa.seasonal_decompose(train_data['Sales'], model='Additive')
# fig = decomposition.plot()
# plt.show()
# plt.figure(figsize=(12, 8))
# plt.subplot(1,2,1)
# sns.catplot(x='Open', kind="count", data=train_data)
# plt.subplot(1,2,2)
# sns.catplot(x='SchoolHoliday', kind="count", data=train_data)
# plt.show()



# plt.figure(figsize=(18,6))
# plt.subplot(1,3,1)
# sns.countplot(x="StateHoliday", hue='Open', data=train_data)
# plt.subplot(1,3,2)
# sns.countplot(x="SchoolHoliday", hue="Open", data=train_data)
# plt.subplot(1,3,3)
# sns.countplot(x="DayOfWeek", hue="Open", data=train_data)
# plt.tight_layout()
# plt.savefig('cat_open')
#
#
# plt.figure(figsize=(16,8))
# plt.subplot(1,2,1)
# sns.countplot(x="Assortment", data=train_data)
# plt.subplot(1,2,2)
# sns.countplot(x="StoreType", data=train_data)
# plt.tight_layout()
# plt.savefig('cat_store')
#
# plt.figure(figsize=(18,6))
# plt.subplot(1,3,1)
# sns.countplot(x="Promo", data=train_data)
# plt.subplot(1,3,2)
# sns.countplot(x="Promo2", data=train_data)
# plt.subplot(1,3,3)
# sns.countplot(x="PromoInterval", data=train_data)
# plt.tight_layout()
# plt.savefig('cat_promo')
#
#
# plt.figure(figsize=(18,6))
# plt.subplot(1,3,1)
# sns.violinplot(x="Promo", y='Sales', data=train_data)
# plt.subplot(1,3,2)
# sns.violinplot(x="Promo2", y='Sales', data=train_data)
# plt.subplot(1,3,3)
# sns.violinplot(x="PromoInterval", y='Sales', data=train_data)
# plt.tight_layout()
# plt.savefig('dis_promo')

train_data = train_data[train_data['Store']=='1']
plt.figure(figsize=(18,6))
plt.subplot(1,3,1)
sns.violinplot(x="StateHoliday", y='Sales', data=train_data)
plt.subplot(1,3,2)
sns.violinplot(x="SchoolHoliday", y='Sales', data=train_data)
plt.subplot(1,3,3)
sns.violinplot(x="DayOfWeek", y='Sales', data=train_data)
plt.tight_layout()
plt.savefig('dis_open')

#
# plt.figure(figsize=(16,8))
# plt.subplot(1,2,1)
# sns.violinplot(x="Assortment", y='Sales', data=train_data)
# plt.subplot(1,2,2)
# sns.violinplot(x="StoreType", y='Sales', data=train_data)
# plt.tight_layout()
# plt.savefig('dis_store')

# plt.figure(figsize=(12,8))
# plt.subplot(2,2,1)
# sns.violinplot(x="Promo", y='Sales', data=train_data)
# plt.subplot(2,2,2)
# sns.violinplot(x="Promo2", y='Sales', data=train_data)
# plt.subplot(2,2,3)
# sns.violinplot(x="PromoInterval", y='Sales', data=train_data)
#
# plt.savefig('dis_promo')

# time_series_plot(train_data)


