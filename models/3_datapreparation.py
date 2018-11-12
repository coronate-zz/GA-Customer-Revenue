import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import time
from pandas.core.common import SettingWithCopyWarning
import warnings
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

from aiqutils.data_preparation import create_categorical_dummies

#df_train = pd.read_csv("../input/df_train-flattened.csv", index=False)
df_train = pd.read_csv("../input/train-flattened.csv")
df_test = pd.read_csv("../input/test-flattened.csv")


df_rolling_past = pd.read_csv("../input/df_rolling_past_godel.csv")
df_rolling_future = pd.read_csv("../input/df_rolling_future_godel.csv")

df_rolling_past["fullVisitorId"] = df_rolling_past.fullVisitorId.astype(float)
df_rolling_future["fullVisitorId"] = df_rolling_future.fullVisitorId.astype(float)
df_rolling_past["date"] = pd.to_datetime(df_rolling_past.date)
df_rolling_future["date"] = pd.to_datetime(df_rolling_future.date)

print(df_rolling_past.shape)
print(df_rolling_future.shape)
df_rolling = df_rolling_past.merge(df_rolling_future, on = ["fullVisitorId", "date"], how ="inner")
print(df_rolling.shape)

del df_rolling_past
del df_rolling_future

df_train["fullVisitorId"] = df_train.fullVisitorId.astype(float)
df_test["fullVisitorId"] = df_test.fullVisitorId.astype(float)
df_train["date"] = pd.to_datetime(df_train.date)
df_test["date"] = pd.to_datetime(df_test.date)

df_train = df_train.groupby(["fullVisitorId", "date"]).sum()
df_train = df_train.reset_index()
df_test = df_test.groupby(["fullVisitorId", "date"]).sum()
df_test = df_test.reset_index()

#raise ValueError("--------------------stop")
print(df_train.shape)
print(df_test.shape)
df_train = df_train.merge(df_rolling, on =["fullVisitorId", "date"])
df_test = df_test.merge(df_rolling, on =["fullVisitorId", "date"])
print(df_train.shape)
print(df_test.shape)











#--------------------------  Getting data from leak ---------------------------------
train_store_1 = pd.read_csv('../input/exported-google-analytics-data/Train_external_data.csv', low_memory=False, skiprows=6, dtype={"Client Id":'str'})
train_store_2 = pd.read_csv('../input/exported-google-analytics-data/Train_external_data_2.csv', low_memory=False, skiprows=6, dtype={"Client Id":'str'})
test_store_1 = pd.read_csv('../input/exported-google-analytics-data/Test_external_data.csv', low_memory=False, skiprows=6, dtype={"Client Id":'str'})
test_store_2 = pd.read_csv('../input/exported-google-analytics-data/Test_external_data_2.csv', low_memory=False, skiprows=6, dtype={"Client Id":'str'})


# Getting VisitId from Google Analytics...
for df in [train_store_1, train_store_2, test_store_1, test_store_2]:
    df["visitId"] = df["Client Id"].apply(lambda x: x.split('.', 1)[1]).astype(np.int64)


# Merge with train/test data
df_train = df_train.merge(pd.concat([train_store_1, train_store_2]), how="left", on="visitId")
df_test = df_test.merge(pd.concat([test_store_1, test_store_2]), how="left", on="visitId")

common_columns = [x for x in df_train.columns if x in df_test.columns]

df_train =  df_train[common_columns + ["totals.transactionRevenue"]]
df_test = df_test[common_columns]
# Drop Client Id
for df in [df_train, df_test]:
    df.drop("Client Id", 1, inplace=True)

print(df_test.fullVisitorId)
def replace_atypical_categories(df_train, df_test, columnname, pct = .01, base_df = "test"):
    """ Replace all categories in a categorical variable whenever the number of 
    observations in the test or train data set is lower than pct percetage of the
    total number of observations.
    The replaced categories are assigned to "other" category. 

    Input:
    ----------
    df_train: train DataFrame
    df_test: test DataFrame
    columnname:  name of the categorical variable whose categories will
    be replaced.
    pct: percetage of number of the observations that will be required for a categorical
    not to be labeled as atypical.
    base_df: The base DataFrame in which the analysis will be done.

    Output:
    ----------
    df_train and df_test with the columnname variable with the new labels.

    """

    if base_df == "test":
        limit  = len(df_test) *pct
        vc = df_test[columnname].value_counts()
    else:
        limit  = len(df_train) *pct
        vc = df_train[columnname].value_counts()
    
    common = vc > limit
    common = set(common.index[common].values)
    print("Set", sum(vc <= limit), columnname, "categories to 'other';", end=" ")
    
    df_train.loc[df_train[columnname].map(lambda x: x not in common), columnname] = 'other'
    df_test.loc[df_test[columnname].map(lambda x: x not in common), columnname] = 'other'
    print("now there are", df_train[columnname].nunique(), "categories in train")
    return df_train, df_test


df_train = df_train.replace(np.nan, 0)
df_test = df_test.replace(np.nan, 0)


#----------------------------- DROP COLUMNS -------------------------------
cols_to_drop =  list()
#Constant columns
const_cols = [c for c in df_train.columns if df_train[c].nunique(dropna=False)==1 ]

#Variables not in df_test but in trai
print("Variables not in df_test but in df_train : ", set(df_train.columns).difference(set(df_test.columns)))

cols_to_drop = const_cols + ['sessionId']
df_train = df_train.drop(cols_to_drop , axis=1)
df_test = df_test.drop(cols_to_drop, axis=1)


print(df_test.fullVisitorId)

#Add date_withseconds features
for df in [df_train, df_test]:
    df['date_withseconds'] = pd.to_datetime(df['visitStartTime'], unit='s')
    df['sess_date_dow']   = df['date_withseconds'].dt.dayofweek
    df['sess_date_hours'] = df['date_withseconds'].dt.hour
    df['sess_date_dom']   = df['date_withseconds'].dt.day



#-----------------Categorical and numerical transformatios-----------------
#Future Variables
df = pd.concat([df_train, df_test])
df.sort_values(['fullVisitorId', 'date_withseconds'], ascending=True, inplace=True)
df['prev_session'] = (df['date_withseconds'] - df[['fullVisitorId', 'date_withseconds']].groupby('fullVisitorId')['date_withseconds'].shift(1)).astype(np.int64) // 1e9 // 60 // 60
df['prev_session_2'] = (df['date_withseconds'] - df[['fullVisitorId', 'date_withseconds']].groupby('fullVisitorId')['date_withseconds'].shift(2)).astype(np.int64) // 1e9 // 60 // 60
df['prev_session_3'] = (df['date_withseconds'] - df[['fullVisitorId', 'date_withseconds']].groupby('fullVisitorId')['date_withseconds'].shift(3)).astype(np.int64) // 1e9 // 60 // 60

df['next_session'] = (df['date_withseconds'] - df[['fullVisitorId', 'date_withseconds']].groupby('fullVisitorId')['date_withseconds'].shift(-1)).astype(np.int64) // 1e9 // 60 // 60
df['next_session_2'] = (df['date_withseconds'] - df[['fullVisitorId', 'date_withseconds']].groupby('fullVisitorId')['date_withseconds'].shift(-2)).astype(np.int64) // 1e9 // 60 // 60
df['next_session_3'] = (df['date_withseconds'] - df[['fullVisitorId', 'date_withseconds']].groupby('fullVisitorId')['date_withseconds'].shift(-3)).astype(np.int64) // 1e9 // 60 // 60

df.sort_index(inplace=True)
train = df[:len(df_train)]
test = df[len(df_train):]

num_cols = ["totals.hits", "totals.pageviews", "visitNumber", "visitStartTime", 'totals.bounces',  'totals.newVisits']    
for col in num_cols:
    df_train[col] = df_train[col].astype(float)
    df_test[col] = df_test[col].astype(float)
    
print(df_test.fullVisitorId)

excluded_features = [ 'sessionId', 'visitId', 'visitStartTime']

categorical_features = [
    _f for _f in df_train.columns
    if (_f not in excluded_features) & (df_train[_f].dtype == 'object')
]

if "fullVisitorId" in categorical_features:
    categorical_features.remove("fullVisitorId")

for cat in categorical_features:
    df_train, df_test = replace_atypical_categories(df_train, df_test, cat, pct = .00075, base_df = "test")

for cat in categorical_features:
    df_train[cat], indexer = pd.factorize(df_train[cat])
    df_test[cat] = indexer.get_indexer(df_test[cat])


df_train = create_categorical_dummies(df_train, categorical_features , 1000)
df_test = create_categorical_dummies(df_test, categorical_features, 1000)
print(df_test.fullVisitorId)


#User-aggregating features:
for feature in ["totals.hits", "totals.pageviews"]:
    info = pd.concat([df_train, df_test]).groupby("fullVisitorId")[feature].mean()
    df_train["usermean_" + feature] = df_train.fullVisitorId.map(info)
    df_test["usermean_" + feature] = df_test.fullVisitorId.map(info)
    
for feature in ["visitNumber"]:
    info = pd.concat([df_train, df_test]).groupby("fullVisitorId")[feature].max()
    df_train["usermax_" + feature] = df_train.fullVisitorId.map(info)
    df_test["usermax_" + feature] = df_test.fullVisitorId.map(info)



df_train.to_csv("../input/df_train_prepared_godel.csv", index = False)
df_test.to_csv("../input/df_test_prepared_godel.csv", index = False)   #<-EXECUTE FEATURE SELECTION
