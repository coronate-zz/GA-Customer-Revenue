

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import time
from pandas.core.common import SettingWithCopyWarning
import warnings
import lightgbm as lgb
from aiqutils.data_preparation import create_categorical_dummies, interact_categorical_numerical
import pickle


def save_obj(obj, name ):
    with open('../obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name ):
    with open('../obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def change_namereverse(df):
    rename_dict = dict()
    for col in df.columns:
        if col == "fullVisitorId" or col == "reverse_date":
            continue
        else:
            rename_col =  col + "_reverse"
            rename_dict[col] = rename_col
    df = df.rename( columns = rename_dict)
    return df


class smart_dict(dict):
    def __missing__(self, key):
        return key

#-----------------------------Upload INFO and get training sets-----------------------------------


numerical_cols = ["totals.hits", "totals.pageviews", "visitNumber", 
                "visitStartTime", 'totals.bounces',  'totals.newVisits', 
                "totals.transactionRevenue"]    

id_columns = ["fullVisitorId"]



#-------------------------------------------------------------------------------------------------

df.to_csv("../input/df_filled_timeseries_reversedate.csv", index = False)
df = pd.read_csv("../input/df_filled_timeseries_reversedate.csv")
reverseddateid_to_ordinaldate = load_obj("reverseddateid_to_ordinaldate")
df = df.sample(1000)
df["date"] = pd.to_datetime(df.date)
df["reverse_date"] = pd.to_datetime(df.reverse_date)
df = df.groupby(["fullVisitorId", "date"]).sum().reset_index()



numerical_cols = ["totals.hits", "totals.pageviews", "visitNumber", 
                "visitStartTime", 'totals.bounces',  'totals.newVisits', 
                "totals.transactionRevenue"]    



df_count = df.groupby(["fullVisitorId"])["visitNumber"].count().reset_index()
df_count =df_count.reset_index()
multiple_visits_clients = list(df_count["fullVisitorId"][df_count["visitNumber"] >1])

#BUG generate dictionary of client dates for te reverse dates.

df_part_A = df[~df.fullVisitorId.isin( multiple_visits_clients)]
df_part_A = df_part_A.reset_index(drop = True)
df_part_A["nonmultiple_dates_col"] = df_part_A["fullVisitorId"].astype(str) + df_part_A["date"].astype(str)

nonmultiple_dates_map = dict()
for id in df_part_A.nonmultiple_dates_col.unique():
    nonmultiple_dates_map[id] = id[-8:]

nonmultiple_dates_map = smart_dict(nonmultiple_dates_map)


new_numerical_cols = list()
#Generación de variables cuadraticas y cubicas
for numerical_col in numerical_cols:
    for power in["square", "cubic", "cuadratic"]:
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
lag_list         = [0,1,2,7]
rolling_list     = [1,2,4,7,14,30]

df_ewm = interact_categorical_numerical(
                                   df, lag_col, numerical_cols,
                                   categorical_cols, lag_list,
                                   rolling_list, agg_funct="sum",
                                   rolling_function = "ewm", freq=None,
                                   group_name=None, store_name=False)

df_rolling = interact_categorical_numerical(
                                   df, lag_col, numerical_cols,
                                   categorical_cols, lag_list,
                                   rolling_list, agg_funct="sum",
                                   rolling_function = "rolling", freq=None,
                                   group_name=None, store_name=False)
df_expansion = interact_categorical_numerical(
                                   df, lag_col, numerical_cols,
                                   categorical_cols, lag_list,
                                   rolling_list, agg_funct="sum",
                                   rolling_function = "expanding", freq=None,
                                   group_name=None, store_name=False)


#Merging new Variables
id_columns = ["fullVisitorId", "date"]
df= df.merge(df_ewm, on = id_columns )
df= df.merge(df_expansion, on = id_columns )
df= df.merge(df_rolling, on = id_columns )

#Processing lag and window functions.
lag_col          = "reverse_date"
categorical_cols = ["fullVisitorId"]
lag_list         = [0,1,2,7]
rolling_list     = [1,2,4,7,14,30]


new_dict = dict()
for key in reverseddateid_to_ordinaldate.keys():
    new_key = key[:-19]
    new_dict[new_key] = str(reverseddateid_to_ordinaldate[key])[:10] 
reverseddateid_to_ordinaldate = smart_dict(new_dict)


df_ewm = interact_categorical_numerical(
                                   df, lag_col, numerical_cols,
                                   categorical_cols, lag_list,
                                   rolling_list, agg_funct="sum",
                                   rolling_function = "ewm", freq=None,
                                   group_name=None, store_name=False)
df_ewm = change_namereverse(df_ewm)
df_ewm["date"] = df_ewm["fullVisitorId"].astype(str) + df_ewm["reverse_date"].astype(str)
df_ewm["date"] = df_ewm["date"].map(reverseddateid_to_ordinaldate)
df_ewm["date"] = df_ewm["date"].map(nonmultiple_dates_map)
df_ewm["date"] = df_ewm["date"].astype(np.datetime64)
del df_ewm["reverse_date"]


df_rolling = interact_categorical_numerical(
                                   df, lag_col, numerical_cols,
                                   categorical_cols, lag_list,
                                   rolling_list, agg_funct="sum",
                                   rolling_function = "rolling", freq=None,
                                   group_name=None, store_name=False)
df_rolling = change_namereverse(df_rolling)
df_rolling["date"] = df_rolling["fullVisitorId"].astype(str) + df_rolling["reverse_date"].astype(str)
df_rolling["date"] = df_rolling["reverse_date"].map(reverseddateid_to_ordinaldate)
df_rolling["date"] = df_rolling["date"].map(nonmultiple_dates_map)
df_rolling["date"] = df_rolling["date"].astype(np.datetime64)
del df_rolling["reverse_date"]

df_expansion = interact_categorical_numerical(
                                   df, lag_col, numerical_cols,
                                   categorical_cols, lag_list,
                                   rolling_list, agg_funct="sum",
                                   rolling_function = "expanding", freq=None,
                                   group_name=None, store_name=False)
df_expansion = change_namereverse(df_expansion)
df_expansion["date"] = df_expansion["fullVisitorId"].astype(str) + df_expansion["reverse_date"].astype(str)
df_expansion["date"] = df_expansion["reverse_date"].map(reverseddateid_to_ordinaldate)
df_expansion["date"] = df_expansion["date"].map(nonmultiple_dates_map)
df_expansion["date"] = df_expansion["date"].astype(np.datetime64)
del df_expansion["reverse_date"]

#Merging new Variables
df= df.merge(df_ewm, on = id_columns )
df= df.merge(df_expansion, on = id_columns )
df= df.merge(df_rolling, on = id_columns )


#Clean Variables:
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

df.to_csv("../input/time_series.csv", index = False)