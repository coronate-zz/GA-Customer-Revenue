import pandas as pd
import numpy as np
import random
import time
import utils_model_genetic 
import pickle
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error 
import importlib 
from sklearn import linear_model
import utils_exomodel
import models 
from  aiqutils.data_preparation import interact_categorical_numerical
from datetime import timedelta


importlib.reload(utils_exomodel)
importlib.reload(models)
importlib.reload(utils_model_genetic)

SKLEARN_MODELS = models.load_models()
models_list =  list(SKLEARN_MODELS.keys())


def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)



#-----------------------------Upload INFO and get training sets-----------------------------------

#df_train = pd.read_csv("../input/df_train-flattened.csv", index=False)

df_train = pd.read_csv("../input/df_train_1prepared_withdate.csv")
df_test = pd.read_csv("../input/df_test_1prepared_withdate.csv")   #<-EXECUTE FEATURE SELECTION
df = pd.concat([df_train, df_test])
df = df.reset_index()
df = df.drop_duplicates()
df = df.replace(np.nan, 0)

df["date"] = pd.to_datetime(df.date)
date_col = "date"
id_columns = ["fullVisitorId"]
fillmethod = "zeros"


#df_test = aiqutils.data_preparation.fill_timeseries(df, id_columns, date_col, freq = "D", fillmethod = "zeros")

lag_col          = "date"
numerical_cols   = ["totals.transactionRevenue"]
categorical_cols = ["fullVisitorId"]
lag_list         = [1,2,3,4,6,8,10,12,16]
rolling_list     = [1,2,3,4,6,8,12,16,20]



def fill_timeseries( df, id_columns, date_col, target_variables, freq = "D", fillmethod = "zeros"):
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
        date must be excluded.
    date_col: str
        Column name for the  date column.
    freq: str
        Frequency od the obseravtions.
            D - if reported daily.
            M - if reported monthly ...
            B   business day frequency
            C   custom business day frequency
            H   hourly frequency
            A, Y year end frequency
            S   secondly frequency
            SM  semi-month end frequency (15th and end of month)             
    fillmethod: str:
        zeros :replace all non reported values with 0.
        mean  :replace all non reported values with the average of the reported values.
        ffill :replace all non reported values using the closest reported value.


    Returns
    -------
    A DataFrame with complete number of observations, filling the nan values 
    with the last value or     0.

    Notes:
    -------
    It's possible to rebuild this function using parallel programming.

    """
    df_result = pd.DataFrame()
    df[date_col] =  pd.to_datetime(df[date_col])    
    df["ID"] = ""
    for col in id_columns:
            df["ID"] = df.ID.apply(str) + "_" + col +df[col].apply(str)
            df[col] = df[col].apply(str)
    for id_variable in df.ID.unique():
        id_columns_date = id_columns + [date_col]
        df_subset = df[df.ID == id_variable]
        df_subset = df_subset.groupby(id_columns_date)[target_variables].sum() #sum or count
        df_subset = df_subset.reset_index()

        start_date = df_subset[date_col].min() #- timedelta(days = daysmax_lag)
        end_date = df_subset[date_col].max() #- timedelta(days = days =max_lag)
        idx=pd.date_range(start=start_date,end=end_date, freq= freq)

        if start_date != end_date:
            print("\n\nMore than one value: \n\t ID: {} \t\n start_date: {} \n\t end_date: {}".format( id_variable, start_date, end_date))

        if fillmethod == "zeros":
            id_values = dict()
            for col in id_columns:
                id_values[col] = df_subset[col].unique()[0]
            df_subset = df_subset.set_index(df_subset[date_col],drop=True)
            df_subset = df_subset.reindex(idx)
            df_subset = df_subset.replace(np.nan, 0)

            for col in id_columns:
                df_subset[col] = id_values[col] 

            df_subset = df_subset.sort_index(ascending=False).drop( date_col,1).reset_index().rename(columns={'index':date_col})

        elif fillmethod == "mean":
            id_values = dict()
            for col in id_columns:
                id_values[col] = df_subset[col].unique()[0]

            df_subset = df_subset.reindex(idx)
            df_subset.replace(nan, 0)

            for col in id_columns:
                df_subset[col] = id_values[col] 
            df_subset.reindex(idx).fillna(df.mean()).sort_index(ascending=False).drop( date_col,1).reset_index().rename(columns={'index':date_col})

        elif fillmethod == "ffil":
            df_subset=df_subset.set_index(df_subset[date_col],drop=True)
            df_subset.reindex(idx).fillna(method='ffill').sort_index(ascending=False).drop( date_col,1).reset_index().rename(columns={'index':date_col})
        else:
            raise Exception("'fllmethod {} is not implemented".format(fillmethod))

        if len(df_result) == 0:
            df_result = df_subset
        else:
            df_result = df_result.append(df_subset)
    return df_result

df["date_perday"] = df.date.apply(lambda x : x.date())
numerical_cols = ["totals.hits", "totals.pageviews", 'totals.transactionRevenue']
df_result = fill_timeseries(df, id_columns, "date_perday", numerical_cols)

"""
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


id_columns = id_columns + ["date"]
df= df.merge(df_ewm, on = id_columns )
df= df.merge(df_expansion, on = id_columns )
df= df.merge(df_rolling, on = id_columns )
"""


print("FINAL")