

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import time
from pandas.core.common import SettingWithCopyWarning
import warnings
import lightgbm as lgb
from aiqutils.data_preparation import create_categorical_dummies, interact_categorical_numerical
import pickle

def fill_timeseries( df, id_columns,  numerical_cols, date_col,  multiple_visits_clients, freq = "D", fillmethod = "zeros"):
    """ Function that fill all the ommited observations for freq reported observations. 

    Process
    -------
    1. for each ID the start and end date is calculated. 
    2. A subset of the df is created using just a single ID.
        2.a An agregation over the id_columns + date_col is performed to avoid
            double indexing problems.
    3. A DataFrame is created susing the complete dates between start_date
       and end_date in intervals of freq time.
    4. The dates DataFrame is used a baseline and it's merge with the subset df by date.
    5. If the are some dates taht are

    Parameters
    ----------
    df: DataFrame
        Data to be processed
    id_columns: list
        List of the variables that identifies each row as a independent value.
        Date must be excluded.
    date_col: str
        Column name for the  date column.
    freq: str
        Frequency od the obseravtions.
            'D' - if reported daily.
            'M' - if reported monthly ...
    fillmethod: str:
        zeros :replace all non reported values with 0.
        mean  :replace all non reported values with the average of the reported values.
        ffill :replace all non reported values using the closest reported value.


    Returns
    -------
    A DataFrame with complete number of observations, filling the nan values 
    with the last value or     0.

    """

    df_part_A = df[~df.fullVisitorId.isin( multiple_visits_clients)]
    df_part_A = df_part_A.reset_index(drop = True)
    df_part_A = df_part_A[numerical_cols + id_columns + [date_col]]


    df_part_B = df[df.fullVisitorId.isin( multiple_visits_clients)]
    df_part_B = df_part_B.reset_index(drop = True)


    df_result = pd.DataFrame()
    df_part_B[date_col] =  pd.to_datetime(df_part_B[date_col])    


    for id_variable in  tqdm(df_part_B.fullVisitorId.unique()):
        id_columns_date = id_columns + [date_col]
        df_subset = df_part_B[df_part_B.fullVisitorId == id_variable]
        if len(df_subset)==0:
          raise ValueError("Empty subset at fill_timeseries")

        df_subset = df_subset[id_columns_date +  numerical_cols].groupby(id_columns_date).sum()
        df_subset = df_subset.reset_index()

        start_date = df_subset[date_col].min() 
        end_date = df_subset[date_col].max()
        if start_date == end_date:
            continue
        else:
            idx=pd.date_range(start=start_date,end=end_date, freq='D')

            id_values = dict()
            for col in id_columns:
                id_values[col] = df_subset[col].unique()[0]

            df_subset = df_subset.set_index(df_subset[date_col],drop=True)
            df_subset = df_subset.reindex(idx)
            df_subset = df_subset.replace(np.nan, 0)

            for col in id_columns:
                df_subset[col] = id_values[col] 

            df_subset = df_subset.sort_index(ascending=False).drop( date_col,1).reset_index().rename(columns={'index':date_col})
        if len(df_subset)==0:
          raise ValueError("Empty subset at fill_timeseries")

        if len(df_result) == 0:
            df_result = df_subset
            res_cols = df_result.columns
        else:
            len_start = len(df_result.fullVisitorId.unique())
            df_result = df_result.append(df_subset[res_cols])
            len_end = len(df_result.fullVisitorId.unique())
            if len_end ==len_start:
              raise ValueError("ID not captured")

    df_result = df_result.append(df_part_A[res_cols])
    df_result = df_result.reset_index()

    return df_result


def  reverse_dates_fun(df, id_columns, date_col,  multiple_visits_clients):

    df_part_A = df[~df.fullVisitorId.isin( multiple_visits_clients)]
    df_part_A = df_part_A.reset_index(drop = True)
    df_part_A["reverse_date"] = df_part_A[date_col]


    df_part_B = df[df.fullVisitorId.isin( multiple_visits_clients)]
    df_part_B = df_part_B.reset_index(drop = True)


    df_result = pd.DataFrame()
    df_part_B[date_col] =  pd.to_datetime(df_part_B[date_col])    

    reverseddateid_to_ordinaldate = dict()
    for id_variable in  tqdm(df_part_B.fullVisitorId.unique()):
        id_columns_date = id_columns + [date_col]
        df_subset = df_part_B[df_part_B.fullVisitorId == id_variable]
        subset_dates = list(df_subset[date_col].unique())
        reverse_order = sorted(subset_dates, reverse = True)
        natural_order = sorted(subset_dates, reverse = False)
        normal_reverse_date = dict()

        normal_reverse_date = dict(zip(natural_order, reverse_order))
        reverse_order_with_id = [str(id_variable) + str(x) for x in reverse_order]
        reverseddateid_to_ordinaldate.update(dict(zip(reverse_order_with_id, natural_order)))

        df_subset["reverse_date"] = df_subset[date_col].map(normal_reverse_date)
        if len(df_result) ==0:
            df_result = df_subset
        else:
            df_result = df_result.append(df_subset)

    df_result = df_result.append(df_part_A)
    df_result = df_result.reset_index()

    df_result.to_csv("../input/df_filled_timeseries_reversedate.csv", index = False)

    return df_result, reverseddateid_to_ordinaldate

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

#df_train = pd.read_csv("../input/df_train-flattened.csv", index=False)
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

numerical_cols = ["totals.hits", "totals.pageviews", "visitNumber", 
                "visitStartTime", 'totals.bounces',  'totals.newVisits', 
                "totals.transactionRevenue"]    

id_columns = ["fullVisitorId"]




df_count = df.groupby(["fullVisitorId"])["visitNumber"].count().reset_index()
df_count =df_count.reset_index()
multiple_visits_clients = list(df_count["fullVisitorId"][df_count["visitNumber"] >1])



#Create new column with dates  reverse_dates_fun

df = fill_timeseries( df, id_columns,  numerical_cols, "date",  multiple_visits_clients =  multiple_visits_clients, freq = "D", fillmethod = "zeros")
df.to_csv("../input/df_filled_timeseries.csv")
df = pd.read_csv("../input/df_filled_timeseries.csv"))
df, reverseddateid_to_ordinaldate  =  reverse_dates_fun(df, id_columns, "date",  multiple_visits_clients )

