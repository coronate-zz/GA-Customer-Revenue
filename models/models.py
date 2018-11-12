import pandas as pd 
import numpy as np 
import multiprocessing
from xgboost import XGBRegressor
from xgboost import plot_importance
import utils_model_genetic 
import pickle
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error 
import utils_exomodel 
import lightgbm as lgb 
from sklearn.model_selection  import train_test_split
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, \
                             BaggingRegressor, ExtraTreesRegressor, \
                             GradientBoostingRegressor
import utils_model_genetic

from sklearn.linear_model import LinearRegression, BayesianRidge,ElasticNet, ARDRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
import importlib 
importlib.reload(utils_model_genetic)
importlib.reload(utils_exomodel)


N =1


def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

class TEST_PARALLEL():
    """docstring for ClassName"""
    def __init__(self, model, N):
        self.model = ""
        self.cores_number = int(np.ceil(multiprocessing.cpu_count()/N))
        print(" TEST MODEL ")

    def fit(self, X_train, y_train, X_test, y_test):
        self.hash_columns =hash("".join([str(x) for x in list(X_train.columns)]))

    def predict(self, X_test):
        #print(self.hash_columns)
        return(self.hash_columns)


class LGBM():
    """docstring for ClassName"""
    def __init__(self, lgb, N):
        self.model = ""
        self.cores_number = int(np.ceil(multiprocessing.cpu_count()/N))
        print("Lightgbm Cores: ")

    def fit(self, X_train, y_train, X_test, y_test,error_type = "MAE"):

        error_dict = {"MSE":"l2", "R2":{"l1","l2"}, "MAE":"l1","LOGLOSS": "multi_logloss" }
        error_metric = error_dict[error_type]

        self.model= lgb.LGBMRegressor(
                        num_leaves=31,
                        learning_rate=0.07,
                        n_estimators=100,
                        subsample=.9,
                        colsample_bytree=.9,
                        random_state=1 )
        self.model.fit(
                X_train, np.log1p(y_train),
                eval_set=[(X_test, np.log1p(y_test))],
                early_stopping_rounds=20,
                verbose=100,
                eval_metric=error_metric)

    def predict(self, X_test):
         prediction=self.model.predict(X_test)
         return(prediction)






class ADABOOST():
    """docstring for ClassName"""
    def __init__(self, AdaBoostRegressor, N):
        self.cores_number = int(np.ceil(multiprocessing.cpu_count()/N))
        self.model = AdaBoostRegressor(
            base_estimator=None, 
            learning_rate=1.0, 
            loss='linear',
            n_estimators=50, 
            random_state=None)

        print("AdaBoostRegressor Cores: ", np.nan)

    def fit(self, X_train, y_train, X_test, y_test, error_type = "MAE"):

        error_dict = {"MSE":"rmse", "R2":{"l1","l2"}, "MAE":"mae","LOGLOSS": "multi_logloss" }
        error_metric = error_dict[error_type]
        self.model.fit(X_train, y_train )

    def predict(self, X_test):
         prediction=self.model.predict(X_test)
         return(prediction)



class EXTRATREE():
    """docstring for ClassName"""
    def __init__(self, ExtraTreesRegressor, N):
        self.cores_number = int(np.ceil(multiprocessing.cpu_count()/N))

        self.model = ExtraTreesRegressor(
                      bootstrap=False, 
                      criterion='mse', 
                      max_depth=None,
                      max_features='auto', 
                      max_leaf_nodes=None,
                      min_impurity_decrease=0.0, 
                      min_impurity_split=None,
                      min_samples_leaf=1, 
                      min_samples_split=2,
                      min_weight_fraction_leaf=0.0, 
                      n_estimators=200, 
                      n_jobs=self.cores_number,
                      oob_score=False, 
                      random_state=None, 
                      verbose= True, 
                      warm_start=False)


        print("ExtraTreesRegressor Cores: ", self.cores_number)

    def fit(self, X_train, y_train, X_test, y_test, error_type = "MAE"):

        error_dict = {"MSE":"mse", "MAE":"mae" }
        error_metric = error_dict[error_type]
        self.model.set_params(criterion = error_metric)
        self.model.fit(X_train, y_train )

    def predict(self, X_test):
         prediction=self.model.predict(X_test)
         return(prediction)



class XGBOOST():
    """docstring for ClassName"""
    def __init__(self, XGBRegressor, N):
        self.model = XGBRegressor
        self.cores_number = int(np.ceil(multiprocessing.cpu_count()/N))
        print("XGBoostRegressor Cores: ", self.cores_number )

    def fit(self, X_train, y_train, X_test, y_test, error_type = "MAE"):

        error_dict = {"MSE":"rmse", "R2":{"l1","l2"}, "MAE":"mae","LOGLOSS": "multi_logloss" }
        error_metric = error_dict[error_type]

        self.model = XGBRegressor(max_depth=6,  #Equals to no limit equivalent to None sklearn
                        learning_rate=0.075, 
                        n_estimators=100, 
                        silent=True, 
                        objective='reg:linear', 
                        nthread=1, 
                        gamma=0,
                        min_child_weight=1, 
                        max_delta_step=0, 
                        subsample=0.85, 
                        colsample_bytree=0.7, 
                        colsample_bylevel=1, 
                        reg_alpha=0, 
                        reg_lambda=1,   
                        scale_pos_weight=1, 
                        seed=1440, 
                        missing=None)

        self.model.fit(X_train, np.log1p(y_train), 
            eval_metric=error_metric, verbose = False, 
            eval_set = [(X_train, np.log1p(y_train)), (X_test, np.log1p(y_test))], 
            early_stopping_rounds=20)


    def predict(self, X_test):
         prediction=self.model.predict(X_test)
         return(prediction)




class BAGGING():
    """docstring for ClassName"""
    def __init__(self, BaggingRegressor, N):
        self.cores_number = int(np.ceil(multiprocessing.cpu_count()/N))
        self.model = BaggingRegressor(
                 base_estimator=None, 
                 bootstrap=True,
                 bootstrap_features=False, 
                 max_features=1.0, 
                 max_samples=1.0,
                 n_estimators=10, 
                 n_jobs= self.cores_number, 
                 oob_score=False, 
                 random_state=None,
                 verbose=0, 
                 warm_start=False)


        print("Bagging Cores: ", self.cores_number)

    def fit(self, X_train, y_train, X_test, y_test, error_type = "MAE"):

        error_dict = {"MSE":"rmse", "R2":{"l1","l2"}, "MAE":"mae","LOGLOSS": "multi_logloss" }
        error_metric = error_dict[error_type]
        self.model.fit(X_train, y_train )

    def predict(self, X_test):
         prediction=self.model.predict(X_test)
         return(prediction)



class LINEARREGRESSION():
    """docstring for ClassName"""
    def __init__(self, LinearRegression, N):
        self.cores_number = int(np.ceil(multiprocessing.cpu_count()/N))
        self.model = LinearRegression(n_jobs = self.cores_number, normalize = False)
        print("LinearRegression Cores: ", self.cores_number )

    def fit(self, X_train, y_train, X_test, y_test, error_type = "MAE"):

        error_dict = {"MSE":"rmse", "R2":{"l1","l2"}, "MAE":"mae","LOGLOSS": "multi_logloss" }
        error_metric = error_dict[error_type]

        self.model.fit(X_train, y_train )

    def predict(self, X_test):
         prediction=self.model.predict(X_test)
         return(prediction)





class ARDR():
    """docstring for ClassName"""
    def __init__(self, ARDRegression, N):
        self.cores_number = int(np.ceil(multiprocessing.cpu_count()/N))
        self.selected_columns = []
        self.model = ARDRegression(
                        alpha_1=1e-06, 
                        alpha_2=1e-06, 
                        compute_score=False, 
                        copy_X=True,
                        fit_intercept=True, 
                        lambda_1=1e-06, 
                        lambda_2=1e-06, 
                        n_iter=300,
                        normalize=False, 
                        threshold_lambda=10000.0, 
                        tol=0.001, verbose=False)


        print("ARDRegression Cores: ", np.nan)

    def fit(self, X_train, y_train, X_test, y_test, error_type = "MAE"):
        try:
            self.selected_columns = np.random.choice(X_train.columns, 100, replace = False)
            X_train = X_train[self.selected_columns]
        except Exception as E:
            X_train = X_train
              
        error_dict = {"MSE":"rmse", "R2":{"l1","l2"}, "MAE":"mae","LOGLOSS": "multi_logloss" }
        error_metric = error_dict[error_type]
        self.model.fit(X_train, y_train )

    def predict(self, X_test):
         prediction=self.model.predict(X_test[self.selected_columns])
         return(prediction)




class BAYESIANRIDGE():
    """docstring for ClassName"""
    def __init__(self, BayesianRidge, N):
        self.cores_number = int(np.ceil(multiprocessing.cpu_count()/N))

        self.model = BayesianRidge(
                        alpha_1=1e-06, 
                        alpha_2=1e-06, 
                        compute_score=False, 
                        copy_X=True,
                        fit_intercept=True, 
                        lambda_1=1e-06, 
                        lambda_2=1e-06, 
                        n_iter=300,
                        normalize=False, 
                        tol=0.001, 
                        verbose=False)


        print("BayesianRidge Cores: ", np.nan)

    def fit(self, X_train, y_train, X_test, y_test, error_type = "MAE"):

        error_dict = {"MSE":"rmse", "R2":{"l1","l2"}, "MAE":"mae","LOGLOSS": "multi_logloss" }
        error_metric = error_dict[error_type]
        self.model.fit(X_train, y_train )

    def predict(self, X_test):
         prediction=self.model.predict(X_test)
         return(prediction)




class ELASTIC():
    """docstring for ClassName"""
    def __init__(self, ElasticNet, N):
        self.cores_number = int(np.ceil(multiprocessing.cpu_count()/N))

        self.model = ElasticNet(
                        alpha=1.0, 
                        copy_X=True, 
                        fit_intercept=True, 
                        l1_ratio=0.5,
                        max_iter=1000, 
                        normalize=False, 
                        positive=False, 
                        precompute=False,
                        random_state=None, 
                        selection='cyclic', 
                        tol=0.0001, 
                        warm_start=False)

        print("ElasticNet Cores: ", np.nan)

    def fit(self, X_train, y_train, X_test, y_test, error_type = "MAE"):

        error_dict = {"MSE":"rmse", "R2":{"l1","l2"}, "MAE":"mae","LOGLOSS": "multi_logloss" }
        error_metric = error_dict[error_type]
        self.model.fit(X_train, y_train )

    def predict(self, X_test):
         prediction=self.model.predict(X_test)
         return(prediction)





class RANDOMFOREST():
    """docstring for ClassName"""
    def __init__(self, RandomForestRegressor, N):
        self.cores_number = int(np.ceil(multiprocessing.cpu_count()/N))

        self.model = RandomForestRegressor(
                       bootstrap=True, criterion='mse',
                       max_depth=None,
                       max_features='auto',
                       max_leaf_nodes=None,
                       min_impurity_decrease=0.0,
                       min_impurity_split=None,
                       min_samples_leaf=1,
                       min_samples_split=2,
                       min_weight_fraction_leaf=0.0,
                       n_estimators=200, 
                       n_jobs=1,
                       oob_score=False, 
                       random_state=None,
                       verbose= True,
                       warm_start=False)


        print("RandomForestRegressor Cores: ", self.cores_number )

    def fit(self, X_train, y_train, X_test, y_test, error_type = "MAE"):

        error_dict = {"MSE":"rmse", "R2":{"l1","l2"}, "MAE":"mae","LOGLOSS": "multi_logloss" }
        error_metric = error_dict[error_type]

        self.model.fit(X_train, y_train )

    def predict(self, X_test):
         prediction=self.model.predict(X_test)
         return(prediction)





class GRADIENTBOOSTING():
    """This mdoel is extremly expensive since it does't allow apralellization"""
    def __init__(self, GradientBoostingRegressor, N):
        self.cores_number = int(np.ceil(multiprocessing.cpu_count()/N))

        self.model = GradientBoostingRegressor(
                         criterion='friedman_mse', 
                         init=None,
                         learning_rate=0.1, 
                         max_depth=3,  #test None
                         max_features=None,
                         max_leaf_nodes=None, 
                         min_impurity_decrease=0.0,
                         min_impurity_split=None, 
                         min_samples_leaf=1,
                         min_samples_split=2, 
                         min_weight_fraction_leaf=0.0,
                         n_estimators=200,  #test 100, 50
                         presort='auto', 
                         random_state=None,
                         subsample=1.0, 
                         verbose=True, 
                         warm_start=False)

        print("GradientBoostingRegressor Cores: ", self.cores_number)

    def fit(self, X_train, y_train, X_test, y_test, error_type = "MAE"):

        error_dict = {"MSE":"mse", "MAE":"mae" }
        error_metric = error_dict[error_type]
        #self.model.set_params(criterion = error_metric)
        self.model.fit(X_train, y_train )

    def predict(self, X_test):
         prediction=self.model.predict(X_test)
         return(prediction)





class MULTINOMIALNB():
    """docstring for ClassName"""
    def __init__(self, MultinomialNB, N):
        self.cores_number = int(np.ceil(multiprocessing.cpu_count()/N))
        self.model = MultinomialNB(
                        alpha=1.0, 
                        class_prior=None, 
                        fit_prior=True)


        print("MultinomialNB Cores: ", np.nan)

    def fit(self, X_train, y_train, X_test, y_test, error_type = "MAE"):
        error_dict = {"MSE":"rmse", "R2":{"l1","l2"}, "MAE":"mae","LOGLOSS": "multi_logloss" }
        error_metric = error_dict[error_type]
        self.model.fit(X_train, y_train )

    def predict(self, X_test):
         prediction=self.model.predict(X_test)
         return(prediction)




class GAUSSIANNB():
    """docstring for ClassName"""
    def __init__(self, GaussianNB, N):
        self.cores_number = int(np.ceil(multiprocessing.cpu_count()/N))
        self.model = GaussianNB(priors=None)

        print("GaussianNB Cores: ", np.nan)

    def fit(self, X_train, y_train, X_test, y_test, error_type = "MAE"):
        error_dict = {"MSE":"rmse", "R2":{"l1","l2"}, "MAE":"mae","LOGLOSS": "multi_logloss" }
        error_metric = error_dict[error_type]
        self.model.fit(X_train, y_train )

    def predict(self, X_test):
         prediction=self.model.predict(X_test)
         return(prediction)


def load_models():
    SKLEARN_MODELS ={
                      "lgbm"             : {"function": utils_model_genetic.score_model, 
                                            "model_class":LGBM(lgb, N), 
                                            "model_name": "lgbm", 
                                            "params" : {"n_workers" :16}},

                      "adaboost"         : { "function": utils_model_genetic.score_model, 
                                             "model_class":ADABOOST(AdaBoostRegressor, N), 
                                             "model_name": "adaboost", 
                                             "params" : {"n_workers" : 16}},

                      "extratrees"       : { "function": utils_model_genetic.score_model, 
                                             "model_class":EXTRATREE(ExtraTreesRegressor, N), 
                                             "model_name": "extratrees", 
                                             "params" : {"n_workers" : 16}},

                      "xgboost"          : { "function": utils_model_genetic.score_model, 
                                              "model_class":XGBOOST(XGBRegressor, N), 
                                             "model_name": "xgboost", 
                                             "params" : {"n_workers" : 16}},

                      "bagging"          : { "function": utils_model_genetic.score_model, 
                                              "model_class":BAGGING(BaggingRegressor, N), 
                                              "model_name": "bagging", 
                                              "params" : {"n_workers": 16}},

                      #"ardr"            : { "function": utils_model_genetic.score_model, 
                      #                     "model_class":ARDR(ARDRegression, N), 
                      #                     "model_name": "", 
                      #                     "params" : {"n_workers" = 16}

                      "bayesridge"       : { "function": utils_model_genetic.score_model, 
                                            "model_class":BAYESIANRIDGE(BayesianRidge, N), 
                                              "model_name": "bayesridge", 
                                              "params" : {"n_workers":  16}},

                      "elasticnet"       : { "function": utils_model_genetic.score_model, 
                                            "model_class":ELASTIC(ElasticNet, N), 
                                             "model_name": "elasticnet", 
                                             "params" : {"n_workers" : 16}},

                      "randomforest"     : { "function": utils_model_genetic.score_model, 
                                             "model_class":RANDOMFOREST(RandomForestRegressor, N), 
                                             "model_name": "randomforest", 
                                             "params" : {"n_workers": 16}},

                      "gradientboosting" : { "function": utils_model_genetic.score_model, 
                                             "model_class":GRADIENTBOOSTING(GradientBoostingRegressor, N), 
                                             "model_name": "gradientboosting", 
                                             "params" : {"n_workers":16}},

                      "multinomial"      : { "function": utils_model_genetic.score_model, 
                                             "model_class":MULTINOMIALNB(MultinomialNB, N), 
                                             "model_name": "multinomial", 
                                             "params" : {"n_workers" :16}},

                      "gaussiannb"       : { "function": utils_model_genetic.score_model, 
                                             "model_class":GAUSSIANNB(GaussianNB, N), 
                                             "model_name": "gaussiannb", 
                                             "params" : {"n_workers": 16}},

                      "linear"           : { "function": utils_model_genetic.score_model, 
                                             "model_class":LINEARREGRESSION(LinearRegression, N),
                                             "model_name": "linear", 
                                             "params" : {"n_workers": 16}},

                      "test_parallel"     : { "function": utils_model_genetic.score_model, 
                                             "model_class":TEST_PARALLEL( "", N),
                                             "model_name": "test_parallel", 
                                             "params" : {"n_workers": 16}}        }
    return SKLEARN_MODELS





#save_obj(SKLEARN_MODELS, "SKLEARN_MODELS")

ERROR_TYPES = {'MAE': mean_absolute_error,
 'MSE': mean_squared_error,
 'R2': r2_score}

save_obj(ERROR_TYPES, "ERROR_TYPES")




def main():

    N = 1
    df = pd.read_csv('/home/coronado/Desktop/clustnet/Data/df_elglobo_sample.csv')
    df["Product_ID"] = df.Product_ID.apply(lambda x:"producto_" + str(x))
    df["Agencia_ID"] = df.Agencia_ID.apply(lambda x:"agencia_" + str(x))
    df = df[["Product_ID", "Agencia_ID","PrecioBruto", "VentaUnidades"]]
    df = utils_exomodel.transform_categorical(df)
    y  = df["VentaUnidades"]
    del df["VentaUnidades"]
    X = df
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


    N =1
    mnb = MULTINOMIALNB(MultinomialNB, N)
    mnb.fit(X_train, y_train, X_test, y_test, "MSE")
    prediction = mnb.predict(X_test)
    print(np.mean(y_test - prediction))


    N =1
    gnb = GAUSSIANNB(GaussianNB, N)
    gnb.fit(X_train, y_train, X_test, y_test, "MSE")
    prediction = gnb.predict(X_test)
    print(np.mean(y_test - prediction))



    lgbm = LGBM(lgb, N)
    lgbm.fit(X_train, y_train, X_test, y_test, "MAE")
    prediction = lgbm.predict(X_test)

    N =1
    ada = ADABOOST(AdaBoostRegressor, N)
    ada.fit(X_train, y_train, X_test, y_test, "MSE")
    prediction = ada.predict(X_test)
    print(np.mean(y_test - prediction))

    N =1

    etr = EXTRATREE(ExtraTreesRegressor, N)
    etr.fit(X_train, y_train, X_test, y_test, "MSE")
    prediction = etr.predict(X_test)
    print(np.mean(y_test - prediction))


    N =1
    xgboost = XGBOOST(XGBRegressor, N)
    xgboost.fit(X_train, y_train, X_test, y_test, "MSE")
    prediction = xgboost.predict(X_test)

    N =1
    br = BAGGING(BaggingRegressor, N)
    br.fit(X_train, y_train, X_test, y_test, "MSE")
    prediction = br.predict(X_test)
    print(np.mean(y_test - prediction))

    N =1
    lr = LINEARREGRESSION(LinearRegression, N)
    lr.fit(X_train, y_train, X_test, y_test, "MSE")
    prediction = lr.predict(X_test)
    #N =1

    #ardr = ARDR(ARDRegression, N)
    #ardr.fit(X_train, y_train, X_test, y_test, "MSE")
    #prediction = ardr.predict(X_test)
    #print(np.mean(y_test - prediction))

    N =1
    bayesrid = BAYESIANRIDGE(BayesianRidge, N)
    bayesrid.fit(X_train, y_train, X_test, y_test, "MSE")
    prediction = bayesrid.predict(X_test)
    print(np.mean(y_test - prediction))

    N =1
    elastic = ELASTIC(ElasticNet, N)
    elastic.fit(X_train, y_train, X_test, y_test, "MSE")
    prediction = elastic.predict(X_test)
    print(np.mean(y_test - prediction))

    N =1
    randomforest = RANDOMFOREST(RandomForestRegressor, N)
    randomforest.fit(X_train, y_train, X_test, y_test, "MSE")
    prediction = lr.predict(X_test)

    N =1
    mnb = GRADIENTBOOSTING(GradientBoostingRegressor, N)
    mnb.fit(X_train, y_train, X_test, y_test, "MSE")
    prediction = mnb.predict(X_test)
    print(np.mean(y_test - prediction))

if __name__ == "__main__":
    main()