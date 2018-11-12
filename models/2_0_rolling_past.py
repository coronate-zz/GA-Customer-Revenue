

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import time
from pandas.core.common import SettingWithCopyWarning
import warnings
import lightgbm as lgb
from aiqutils.data_preparation import create_categorical_dummies, interact_categorical_numerical
import pickle


def clean_dataset(df):
    target_lag_variables  = [x for x in df.columns if  "totals.transactionRevenue" in x]
    target_lag_variables_lag0 = [x for x in target_lag_variables if "lag0"  in x]
    df = df.drop(target_lag_variables_lag0, axis =1)

    #roll1 and lag0 has the value of the numerical col
    repeated_columns  = [x for x in df.columns if  "roll1_lag0" in x]
    df = df.drop(repeated_columns, axis =1)
    df = df.dropna(axis=1, how='all')
    df = df.replace(np.nan, 0)

    #Drop single value cols
    for col in df.columns:
        if len(df[col].unique()) == 1:
            df.drop(col,inplace=True,axis=1)
    return df

#-------------------------------------------------------------------------------------------------
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
df["fullVisitorId"] = df["fullVisitorId"].astype(float)
df = df.groupby(["fullVisitorId", "date"]).sum()
df = df.reset_index()

df_timeseries = pd.read_csv("../input/df_filled_timeseries.csv")
df_timeseries["date"] = pd.to_datetime(df_timeseries.date )
df_timeseries["fullVisitorId"] = df_timeseries["fullVisitorId"].astype(str)


numerical_cols = ["totals.hits", "totals.pageviews", "visitNumber", 
                "visitStartTime", 'totals.bounces',  'totals.newVisits', 
                "totals.transactionRevenue"]    



#df_test = df.merge(df_timeseries, on =["fullVisitorId", "date"], how= "left", indicator = True)
#df_test2  = df_test[df_test._merge == "left_only"]


df_count = df_timeseries.groupby(["fullVisitorId"])["visitNumber"].count().reset_index()
df_count =df_count.reset_index()
multiple_visits_clients = list(df_count["fullVisitorId"][df_count["visitNumber"] >1])


new_numerical_cols = list()
#Generación de variables cuadraticas y cubicas
for numerical_col in numerical_cols:
    for power in["square", "cubic"]:
        if power == "square":
            name = numerical_col + "_" + power
            df_timeseries[name] = np.power(df_timeseries[numerical_col],2)
        elif power == "cubic":
            name = numerical_col + "_" + power
            df_timeseries[name] = np.power(df_timeseries[numerical_col], 3)
        elif power == "cuadratic":
            name = numerical_col + "_" + power
            df_timeseries[name] = np.power(df_timeseries[numerical_col] ,4)
        new_numerical_cols.append(name)
numerical_cols =  numerical_cols + new_numerical_cols
print(numerical_cols )


#Processing lag and window functions.
lag_col          = "date"
categorical_cols = ["fullVisitorId"]
lag_list         = [0,1,3,7]
rolling_list     = [1,3,7,14]

df_ewm = interact_categorical_numerical(
                                   df_timeseries, lag_col, numerical_cols,
                                   categorical_cols, lag_list,
                                   rolling_list, agg_funct="sum",
                                   rolling_function = "ewm", freq=None,
                                   group_name=None, store_name=False)
df_ewm = clean_dataset(df_ewm)
df_ewm = df_ewm.replace(np.nan, 0)
print("TEST: ", len(df))
df_ewm["fullVisitorId"] = df_ewm["fullVisitorId"].astype(float)
df = df.merge(df_ewm, on = ["fullVisitorId", "date"], how = "inner")
print("TEST: ", len(df))
del df_ewm

df.to_csv("../input/df_rolling_past.csv")


#Processing lag and window functions.
lag_col          = "date"
categorical_cols = ["fullVisitorId"]
lag_list         = [0,1,3,7]
rolling_list     = [1,4,8,15]

df_rolling = interact_categorical_numerical(
                                   df_timeseries, lag_col, numerical_cols,
                                   categorical_cols, lag_list,
                                   rolling_list, agg_funct="sum",
                                   rolling_function = "rolling", freq=None,
                                   group_name=None, store_name=False)
df_rolling = clean_dataset(df_rolling)
df_rolling = df_rolling.replace(np.nan, 0)
print("TEST: ", len(df.shape))
df_rolling["fullVisitorId"] = df_rolling["fullVisitorId"].astype(float)
df = df.merge(df_rolling, on = ["fullVisitorId", "date"], how = "inner")
print("TEST: ", len(df.shape))
del df_rolling

df.to_csv("../input/df_rolling_past.csv")

#Processing lag and window functions.
lag_col          = "date"
categorical_cols = ["fullVisitorId"]
lag_list         = [0,1,3,7]
rolling_list     = [1,2,7,12]

df_expansion = interact_categorical_numerical(
                                   df_timeseries, lag_col, numerical_cols,
                                   categorical_cols, lag_list,
                                   rolling_list, agg_funct="sum",
                                   rolling_function = "expanding", freq=None,
                                   group_name=None, store_name=False)
df_expansion = clean_dataset(df_expansion)
df_expansion = df_expansion.replace(np.nan, 0)
print("TEST: ", len(df))
df_expansion["fullVisitorId"] = df_expansion["fullVisitorId"].astype(float)
df = df.merge(df_expansion, on = ["fullVisitorId", "date"], how = "inner")
print("TEST: ", len(df))
del df_expansion

del df_timeseries

df.to_csv("../input/df_rolling_past_godel.csv")
