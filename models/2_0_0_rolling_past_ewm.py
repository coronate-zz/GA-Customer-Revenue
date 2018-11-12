

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import time
from pandas.core.common import SettingWithCopyWarning
import warnings
import lightgbm as lgb
from aiqutils.data_preparation import create_categorical_dummies, interact_categorical_numerical
import pickle


#-------------------------------------------------------------------------------------------------


df = pd.read_csv("../input/df_filled_timeseries.csv")
df["date"] = pd.to_datetime(df.date)
df["fullVisitorId"] = df["fullVisitorId"].astype(str)


numerical_cols = ["totals.hits", "totals.pageviews", "visitNumber", 
                "visitStartTime", 'totals.bounces',  'totals.newVisits', 
                "totals.transactionRevenue"]    



df_count = df.groupby(["fullVisitorId"])["visitNumber"].count().reset_index()
df_count =df_count.reset_index()
multiple_visits_clients = list(df_count["fullVisitorId"][df_count["visitNumber"] >1])


new_numerical_cols = list()
#Generación de variables cuadraticas y cubicas
for numerical_col in numerical_cols:
    for power in["square", "cubic"]:
        if power == "square":
            name = numerical_col + "_" + power
            df[name] = np.power(df[numerical_col],2)
        elif power == "cubic":
            name = numerical_col + "_" + power
            df[name] = np.power(df[numerical_col], 3)
        elif power == "cuadratic":
            name = numerical_col + "_" + power
            df[name] = np.power(df[numerical_col] ,4)
        new_numerical_cols.append(name)
numerical_cols =  numerical_cols + new_numerical_cols
print(numerical_cols )


#Processing lag and window functions.
lag_col          = "date"
categorical_cols = ["fullVisitorId"]
lag_list         = [0,1,3,7]
rolling_list     = [1,2,4,7,14]

df_ewm = interact_categorical_numerical(
                                   df, lag_col, numerical_cols,
                                   categorical_cols, lag_list,
                                   rolling_list, agg_funct="sum",
                                   rolling_function = "ewm", freq=None,
                                   group_name=None, store_name=False)


df_train = pd.read_csv("../input/train-flattened.csv")
df_test = pd.read_csv("../input/test-flattened.csv")
df_test["totals.transactionRevenue"] = 0
common_columns = [x for x in df_train.columns if x in df_test.columns]
df_train =  df_train[common_columns ]
df_test = df_test[common_columns]
df = df_train.append(df_test)
df = df.drop_duplicates()
df = df.replace(np.nan, 0)
df["date"] = pd.to_datetime(df.date, format= "%Y%m%d")
df["fullVisitorId"] = df["fullVisitorId"].astype(str)
df = df.groupby(["fullVisitorId", "date"]).sum()
df = df.reset_index()
print("TEST: ", len(df.shape))
df = df.merge(df_ewm, on = ["fullVisitorId", "date"], how = "inner")
print("TEST: ", len(df.shape))

df.to_csv("../input/df_ewm_past.csv")