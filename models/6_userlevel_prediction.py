import pandas as pd
import numpy as np
import gc
import lightgbm as lgb
from tqdm import tqdm
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error

prediction_train_files = [
        "../input/df_train_session_xgboost.csv", 
        "../input/df_train_session_lgbm.csv",
        "../input/df_train_session_linear.csv",
        "../input/df_train_session_randomforest.csv"]

prediction_test_files = [
        "../input/df_train_session_xgboost.csv", 
        "../input/df_train_session_lgbm.csv",
        "../input/df_train_session_linear.csv",
        "../input/df_train_session_randomforest.csv"]


#df_train = pd.read_csv("../input/df_train-flattened.csv", index=False)
selected_features = pd.read_csv("../input/selected_features_persession.csv")
selected_features = list(selected_features["selected_features"])
df_train = pd.read_csv("../input/df_train_prepared.csv")
print(df_train.shape)
df_train = df_train.sample(1000, random_state = 100)
df_test =  pd.read_csv("../input/df_test_prepared.csv")   #<-EXECUTE FEATURE SELECTION

y_reg = df_train['totals.transactionRevenue'].fillna(0)

df_train = df_train[selected_features]
df_test  = df_test[selected_features]

for file in prediction_train_files:
    df_prediction = pd.read_csv(file)
    name =[x for x in df_prediction.columns if "predictions" in x][0]
    df_train[name] = df_prediction[name]

for file in prediction_test_files:
    df_prediction = pd.read_csv(file)
    name =[x for x in df_prediction.columns if "predictions" in x][0]
    df_test[name] = df_prediction[name]

"""
fullVisitorId_list = list(df_test["fullVisitorId"].unique())
map_fullvisitor = dict()
for element in fullVisitorId_list:
    map_fullvisitor[float(element)] =  element
"""
df_test["fullVisitorId"] = df_test["fullVisitorId"].astype(str).astype(float)
df_train["fullVisitorId"] = df_train["fullVisitorId"].astype(str).astype(float) #Get session target

# Aggregate data at User level
train_data = df_train.groupby('fullVisitorId').mean()

# Create a list of predictions for each Visitor
train_predictions_perid = df_train[['fullVisitorId', 'predictions']].groupby('fullVisitorId')\
    .apply(lambda df: list(df.predictions))\
    .apply(lambda x: {'pred_'+str(i): pred for i, pred in enumerate(x)})	

# Create a DataFrame with VisitorId as index
# train_predictions_perid contains dict 
# so creating a dataframe from it will expand dict values into columnss
train_prediction_eachcol = pd.DataFrame(list(train_predictions_perid.values), index=train_data.index)
prediction_feats = train_prediction_eachcol.columns
train_prediction_eachcol['t_mean'] = np.log1p(train_prediction_eachcol[prediction_feats].mean(axis=1))
train_prediction_eachcol['t_median'] = np.log1p(train_prediction_eachcol[prediction_feats].median(axis=1))
train_prediction_eachcol['t_sum_log'] = np.log1p(train_prediction_eachcol[prediction_feats]).sum(axis=1)
train_prediction_eachcol['t_sum_act'] = np.log1p(train_prediction_eachcol[prediction_feats].fillna(0).sum(axis=1))
train_prediction_eachcol['t_nb_sess'] = train_prediction_eachcol[prediction_feats].isnull().sum(axis=1)
full_data = pd.concat([train_data, train_prediction_eachcol], axis=1)
del train_data, train_prediction_eachcol
gc.collect()
full_data.shape


test_predictions_perid = df_test[['fullVisitorId', 'predictions']].groupby('fullVisitorId')\
    .apply(lambda df: list(df.predictions))\
    .apply(lambda x: {'pred_'+str(i): pred for i, pred in enumerate(x)})


test_data = df_test.groupby('fullVisitorId').mean()
test_prediction_eachcol = pd.DataFrame(list(test_predictions_perid.values), index=test_data.index)
for f in prediction_feats:
    if f not in test_prediction_eachcol.columns:
        test_prediction_eachcol[f] = np.nan
test_prediction_eachcol['t_mean'] = np.log1p(test_prediction_eachcol[prediction_feats].mean(axis=1))
test_prediction_eachcol['t_median'] = np.log1p(test_prediction_eachcol[prediction_feats].median(axis=1))
test_prediction_eachcol['t_sum_log'] = np.log1p(test_prediction_eachcol[prediction_feats]).sum(axis=1)
test_prediction_eachcol['t_sum_act'] = np.log1p(test_prediction_eachcol[prediction_feats].fillna(0).sum(axis=1))
test_prediction_eachcol['t_nb_sess'] = test_prediction_eachcol[prediction_feats].isnull().sum(axis=1)
test_full_data = pd.concat([test_data, test_prediction_eachcol], axis=1)
del test_data, test_prediction_eachcol
gc.collect()
test_full_data.shape



df_train['target'] = y_reg
train_userlevel_target = df_train[['fullVisitorId', 'target']].groupby('fullVisitorId').sum()

#Define folding strategy
def get_folds(df=None, n_splits=5):
    """Returns dataframe indices corresponding to Visitors Group KFold"""
    # Get sorted unique visitors
    unique_vis = np.array(df['fullVisitorId'].unique())

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

folds = get_folds(df=full_data[['totals.pageviews']].reset_index(), n_splits=5)

oof_preds = np.zeros(full_data.shape[0])
sub_preds = np.zeros(test_full_data.shape[0])
vis_importances = pd.DataFrame()

for fold_, (trn_, val_) in tqdm(enumerate(folds)):
    print(" -------------- NEW FOLD ---------------------------")
    trn_x, trn_y = full_data.iloc[trn_], train_userlevel_target['target'].iloc[trn_]
    val_x, val_y = full_data.iloc[val_], train_userlevel_target['target'].iloc[val_]
    
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
        eval_set=[(trn_x, np.log1p(trn_y)), (val_x, np.log1p(val_y))],
        eval_names=['TRAIN', 'VALID'],
        early_stopping_rounds=50,
        eval_metric='rmse',
        verbose=100
    )
    imp_df = pd.DataFrame()
    imp_df['feature'] = trn_x.columns
    imp_df['gain'] = reg.booster_.feature_importance(importance_type='gain')
    
    imp_df['fold'] = fold_ + 1
    vis_importances = pd.concat([vis_importances, imp_df], axis=0, sort=False)
    
    oof_preds[val_] = reg.predict(val_x, num_iteration=reg.best_iteration_)
    oof_preds[oof_preds < 0] = 0
    
    # Make sure features are in the same order
    _preds = reg.predict(test_full_data[full_data.columns], num_iteration=reg.best_iteration_)
    _preds[_preds < 0] = 0
    sub_preds += _preds / len(folds)
    
mean_squared_error(np.log1p(train_userlevel_target['target']), oof_preds) ** .5         

class smart_dict(dict):
    def __missing__(self, key):
        return key

map_dict =smart_dict()
map_dict[259678714014] =  '0000000259678714014'  
map_dict[49363351866189] =  '0000049363351866189'  
map_dict[53049821714864] =  '0000053049821714864'  
map_dict[59488412965267] =  '0000059488412965267'  
map_dict[85840370633780] =  '0000085840370633780'  
map_dict[91131414287111] =  '0000091131414287111'  
map_dict[117255350596610] =  '0000117255350596610'  
map_dict[118334805178127] =  '0000118334805178127'  
map_dict[130646294093000] =  '0000130646294093000'  


test_full_data['PredictedLogRevenue'] = sub_preds

sample_submission = pd.read_csv("../input/sample_submission.csv")

for i in tqdm(sample_submission.index):
    ss_id = sample_submission.loc[i, "fullVisitorId"]
    prediction_atid = test_full_data.loc[ float(ss_id) , "PredictedLogRevenue"]
    sample_submission.loc[i, "PredictedLogRevenue"] = prediction_atid
sample_submission.to_csv("../input/test_0.csv", index = False)