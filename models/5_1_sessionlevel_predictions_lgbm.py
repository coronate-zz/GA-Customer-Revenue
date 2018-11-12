import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import time
from pandas.core.common import SettingWithCopyWarning
import warnings
import lightgbm as lgb
from sklearn.model_selection import GroupKFold


#df_train = pd.read_csv("../input/df_train-flattened.csv", index=False)
#selected_features = pd.read_csv("../input/selected_features_persession.csv")
#selected_features = selected_features["selected_features"]
df_train = pd.read_csv("../input/df_train_prepared_godel.csv")
df_test =  pd.read_csv("../input/df_test_prepared_godel.csv")   #<-EXECUTE FEATURE SELECTION

df_train["date"] = pd.to_datetime(df.date)
df_train["reverse_date"] = pd.to_datetime(df.reverse_date)

df_test["date"] = pd.to_datetime(df_test.date)
df_test["reverse_date"] = pd.to_datetime(df_test.reverse_date)


train_dates = df_train["date"]
test_dates = df_test["date"]
del df_test["date"]
del df_train["date"]

#df_test = df_test.sample(5000)

y_reg = df_train['totals.transactionRevenue'].fillna(0)

#df_train = df_train[selected_features]
#df_test  = df_test[selected_features]
df_test["fullVisitorId"] = df_test["fullVisitorId"].astype(str)
df_train["fullVisitorId"] = df_train["fullVisitorId"].astype(str)
#Get session target

#Define folding strategy
def get_folds(df=None, n_splits=5):
    """Returns dataframe indices corresponding to Visitors Group KFold"""
    # Get sorted unique visitors
    unique_vis = df['fullVisitorId'].unique()

    # Get folds
    folds = GroupKFold(n_splits=n_splits)
    fold_ids = []
    ids = np.arange(df.shape[0])
    for trn_vis, val_vis in folds.split(X=unique_vis, y=unique_vis, groups=unique_vis):
        fold_ids.append(
            [
                ids[df['fullVisitorId'].isin(unique_vis[trn_vis])],
                ids[df['fullVisitorId'].isin(unique_vis[val_vis])]
            ]
        )

    return fold_ids


folds = get_folds(df=df_train, n_splits=5)

#-------------------- Predict revenues at session level-----------------


importances = pd.DataFrame()
oof_reg_preds = np.zeros(df_train.shape[0])
sub_reg_preds = np.zeros(df_test.shape[0])

df_test_fullvisitorid_str = df_test["fullVisitorId"].copy()
df_test["fullVisitorId"] = df_test["fullVisitorId"].astype(float)

for fold_, (trn_, val_) in enumerate(folds):
    trn_x, trn_y = df_train.iloc[trn_], y_reg.iloc[trn_]
    val_x, val_y = df_train.iloc[val_], y_reg.iloc[val_]

    trn_x["fullVisitorId"] = trn_x["fullVisitorId"].astype(float)
    val_x["fullVisitorId"] = val_x["fullVisitorId"].astype(float)


    reg = lgb.LGBMRegressor(
        num_leaves=31,
        learning_rate=0.03,
        n_estimators=1000,
        subsample=.9,
        colsample_bytree=.9,
        random_state=1
    )


    reg.fit(
        trn_x, np.log1p(trn_y),
        eval_set=[(val_x, np.log1p(val_y))],
        early_stopping_rounds=50,
        verbose=100,
        eval_metric='rmse'
    )
    imp_df = pd.DataFrame()
    imp_df['feature'] = selected_features
    imp_df['gain'] = reg.booster_.feature_importance(importance_type='gain')
    
    imp_df['fold'] = fold_ + 1
    importances = pd.concat([importances, imp_df], axis=0, sort=False)
    
    oof_reg_preds[val_] = reg.predict(val_x, num_iteration=reg.best_iteration_)
    oof_reg_preds[oof_reg_preds < 0] = 0


    _preds = reg.predict(df_test, num_iteration=reg.best_iteration_)
    _preds[_preds < 0] = 0
    sub_reg_preds += np.expm1(_preds) / len(folds)
    
mean_squared_error(np.log1p(y_reg), oof_reg_preds) ** .5
df_train['predictions_lgbm'] = np.expm1(oof_reg_preds)


pass_columns = ["fullVisitorId", "date", "predictions_lgbm"]

df_train['totals.transactionRevenue'] = y_reg 
df_test['predictions_lgbm'] = sub_reg_preds
df_test["fullVisitorId"] = df_test_fullvisitorid_str
df_test["date"] = test_dates
df_train["date"] = train_dates
df_train[pass_columns].to_csv("../input/df_train_session_lgbm.csv", index = False)
df_test[pass_columns].to_csv("../input/df_test_session_lgbm.csv", index = False)


