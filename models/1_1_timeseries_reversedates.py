

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import time
from pandas.core.common import SettingWithCopyWarning
import warnings
import lightgbm as lgb
from aiqutils.data_preparation import create_categorical_dummies, interact_categorical_numerical
import pickle
from tqdm import tqdm


def  reverse_dates_fun(df, id_columns, date_col,  multiple_visits_clients):

    df_part_A = df[~df.fullVisitorId.isin( multiple_visits_clients)]
    df_part_A = df_part_A.reset_index(drop = True)
    df_part_A["reverse_date"] = df_part_A[date_col]
    reverseddateid_to_ordinaldate_partA = dict( zip(df_part_A["fullVisitorId"].astype(str) + df_part_A["date"].astype(str), df_part_A["date"]))

    df_part_B = df[df.fullVisitorId.isin( multiple_visits_clients)]
    df_part_B = df_part_B.reset_index(drop = True)


    df_result = pd.DataFrame()
    df_part_B[date_col] =  pd.to_datetime(df_part_B[date_col])    

    reverseddateid_to_ordinaldate = dict()
    for id_variable in  tqdm(df_part_B.fullVisitorId.unique()):
        id_columns_date = id_columns + [date_col]
        df_subset = df_part_B[df_part_B.fullVisitorId == id_variable]
        natural_order = df_subset.sort_values(by =["fullVisitorId", "date"],  ascending = True)
        reverse_order = df_subset.sort_values(by =["fullVisitorId", "date"],  ascending = False)
        normal_reverse_date = dict(zip(natural_order["date"], reverse_order["date"]))
        reverseddateid_to_ordinaldate.update(dict(zip(reverse_order["fullVisitorId"].astype(str) + reverse_order["date"].astype(str),
                                                      natural_order["date"])))

        df_subset["reverse_date"] = df_subset[date_col].map(normal_reverse_date)
        if len(df_result) ==0:
            df_result = df_subset
        else:
            df_result = df_result.append(df_subset)

    df_result = df_result.append(df_part_A)
    df_result = df_result.reset_index()

    df_result.to_csv("../input/df_filled_timeseries_reversedate.csv", index = False)
    reverseddateid_to_ordinaldate.update(reverseddateid_to_ordinaldate_partA)
    return df_result, reverseddateid_to_ordinaldate

def save_obj(obj, name ):
    with open('../obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name ):
    with open('../obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


class smart_dict(dict):
    def __missing__(self, key):
        return key

#-----------------------------Upload INFO and get training sets-----------------------------------


numerical_cols = ["totals.hits", "totals.pageviews", "visitNumber", 
                "visitStartTime", 'totals.bounces',  'totals.newVisits', 
                "totals.transactionRevenue"]    

id_columns = ["fullVisitorId"]

#Create new column with dates  reverse_dates_fun
df = pd.read_csv("../input/df_filled_timeseries.csv")
df = df.groupby(["fullVisitorId", "date"]).sum().reset_index()

df_count = df.groupby(["fullVisitorId"])["visitNumber"].count().reset_index()
df_count =df_count.reset_index()
multiple_visits_clients = list(df_count["fullVisitorId"][df_count["visitNumber"] >1])

df, reverseddateid_to_ordinaldate  =  reverse_dates_fun(df, id_columns, "date",  multiple_visits_clients )
df.to_csv("../input/df_filled_timeseries_reversedate.csv", index = False)
save_obj(reverseddateid_to_ordinaldate, "reverseddateid_to_ordinaldate")