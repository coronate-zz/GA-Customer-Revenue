
import pandas as pd 
import numpy as np 
import random 
from   pprint import pprint 
import copy
import pdb, traceback, sys
import multiprocessing
import random
import time
from sklearn import linear_model
import pickle
import threading
import models
import utils_exomodel
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error 
import utils_model_genetic


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

def get_lag8columns(df):
  lag_columns = [x for x in df.columns if "LAG"  in x ]
  exclude_columns = list()
  for i in range(0,8):
    lag_columns_i = [x for x in lag_columns if "LAG" + str(i) in x]
    exclude_columns.extend(lag_columns_i)

  df_columns = set(df.columns) - set(exclude_columns)
  return list(df_columns)

#-----------------------------Upload INFO and get training sets-----------------------------------

df = pd.read_csv('/home/alejandro/Downloads/tender_flake_oscar.csv')
df = df.sample(10000)
df = df.reset_index(drop = True)
lag8columns = get_lag8columns(df)
df = df[lag8columns]
df_copy =  df.copy()

df =  df.drop_duplicates()
df["DESCRIPTION"] = df.DESCRIPTION.apply(lambda x:"producto_" + str(x))
df["MARKET_MEANING"] = df.MARKET_MEANING.apply(lambda x:"agencia_" + str(x))



df["DATE"] = pd.to_datetime(df.DATE, format = "%Y-%m-%d")
df["MONTH"] = df.DATE.apply(lambda x:  x.month)
df["YEAR"] = df.DATE.apply(lambda x:  x.year)



df = df.drop([ 'BRAND', 'CATEGORY_SUBGROUP', 'CB_PRODUCT_SEGMENT', 'DAY', "DATE",
              'MANUFACTURER', 'MARKET', 'MARKET_LEVEL', 'UPC',
              '_Vol', 'HOLIDAY_DUMMY', 'RETAILER', 'Unit_Vol_sqrt',
               'Unit_Vol_cubic', 'TENDERFLAKE_Unit_Vol',
              'COMPLIMENTS_Unit_Vol','SELECTION_Unit_Vol',
              'WESTERN_FAMILY_Unit_Vol','TENDERFLAKE_Unit_Vol_sqrt',
              'COMPLIMENTS_Unit_Vol_sqrt','SELECTION_Unit_Vol_sqrt',
              'WESTERN_FAMILY_Unit_Vol_sqrt','TENDERFLAKE_Unit_Vol_cubic',
              'COMPLIMENTS_Unit_Vol_cubic','SELECTION_Unit_Vol_cubic',
              'WESTERN_FAMILY_Unit_Vol_cubic'], axis =1 )



df["Avg_Retail_Unit_Price"] = (df.Avg_Retail_Unit_Price - df.Avg_Retail_Unit_Price.mean())/df.Avg_Retail_Unit_Price.std()
#df["Unit_Vol"] = (df.Unit_Vol - df.Unit_Vol.mean())/df.Unit_Vol.std()

num_cols = df._get_numeric_data().columns
cat_cols =  [x for x in df.columns if x not in num_cols]

for id_col in cat_cols:
  df_dummies = pd.get_dummies(df[id_col])
  df = df.merge(df_dummies,   right_index = True, left_index = True)

for cat in cat_cols:
  df[cat], indexer = pd.factorize(df[cat])


df = df.replace(np.nan, 0)
df =  df.drop_duplicates()
df_kfolded = utils_exomodel.transform_KFold_random(df, "Unit_Vol", 3 )


M= 4     
N= 4

neuronal_system, chromosome = utils_exomodel.build_neuronal_system(M, N, models_list)
NEURONAL_SOLUTIONS          = utils_exomodel.decode_neuronal_system(neuronal_system, df, SKLEARN_MODELS)
utils_exomodel.get_intramodel_chromosomes(NEURONAL_SOLUTIONS)
#utils_exomodel.simulate_output(NEURONAL_SOLUTIONS, N, df_kfolded)
#df_exmodel = utils_exomodel.get_models_output(NEURONAL_SOLUTIONS, df_kfolded) #runs everytime a layer is finished.

N = 15
PC =.13   #Crossover probability
PM =.13  #Mutation probability
MAX_ITERATIONS = 8

N_WORKERS = 5
round_prediction = False
max_features = 1000

n_col = "N_0"
columns_list =list(df_kfolded["all_data"]["data"].columns)

print("\n\nTEST1 KFOLDED MEANS: ")
for fold in df_kfolded.keys():
    print("FOLD", np.mean(df_kfolded[fold]["y"]))



ERROR_TYPES = {"MAE": mean_absolute_error, "MSE": mean_squared_error, "R2": r2_score}
POPULATION_test = utils_model_genetic.generate_model_population(columns_list, NEURONAL_SOLUTIONS, n_col, N, max_features)
INDIVIDUAL = POPULATION_test[0] 
error_type = "MAE"
MODEL = SKLEARN_MODELS["xgboost"]


print("\n\nTEST2 KFOLDED MEANS: ")
for fold in df_kfolded.keys():
    print("FOLD", np.mean(df_kfolded[fold]["y"]))


t1 = time.time()
POPULATION_X, SOLUTIONS =  utils_model_genetic.solve_genetic_algorithm( N, PC, PM, N_WORKERS, 
                            MAX_ITERATIONS, MODEL, NEURONAL_SOLUTIONS, n_col, columns_list,df_kfolded, 
                            df_exmodel = None , max_features = max_features, round_prediction= False,
                            parallel_execution = False)
t2 = time.time()



#utils_model_genetic.print_SOLUTIONS(SOLUTIONS)
#equal_scores_keys = utils_model_genetic.test_equal_squares(SOLUTIONS)
#utils_model_genetic.test_equal_squares_likelyness(equal_scores_keys, SOLUTIONS)
print("\n\nTIME: {} {}  == {} ".format(t1,t2, t2-t1))

for individual in SOLUTIONS.keys(): 
    print("\n\n-------------NEW SET----------------------------")
    for k in SOLUTIONS.keys():
        same_baseline_features = False
        same_baseline_chromosome = False
        same_genoma = False
        print("    -------   ")
        if SOLUTIONS[individual]["baseline_features"] == SOLUTIONS[k]["baseline_features"]:
            same_baseline_features = True

        if SOLUTIONS[k]["baseline_features_chromosome"] == SOLUTIONS[individual]["baseline_features_chromosome"]:
            same_baseline_chromosome = True 

        if k == individual:
            same_genoma = True

        if same_baseline_chromosome and not same_genoma:
            print("ERROR1: mismo chromosome en bl pero diferente ne genoma total. El chromosoma interno no se ve afectado. Funcion cross_mutate")
        if same_baseline_chromosome and not same_baseline_features:
            print("ERROR2: mismo chromosoma en bl pero diferentes baseline_features ")
        if not same_genoma and same_baseline_features:
            print("ERROR3: mismas variables en baseline_features pero los individuos tienen dieferente genoma")





for individual in SOLUTIONS.keys(): 
    print("\n\n-------------NEW SET----------------------------\n", individual)
    for k in SOLUTIONS.keys():
        if SOLUTIONS[individual]["baseline_features"] == SOLUTIONS[k]["baseline_features"] and SOLUTIONS[k]["genoma"] != SOLUTIONS[individual]["genoma"]:
            print(SOLUTIONS[k]["baseline_features_chromosome"] == SOLUTIONS[individual]["baseline_features_chromosome"])
            print(k)
            #raise ValueError("individual in SOLUTIONS have duplicated fenotype")

test_fenotype_chromosome_SOLUTIONS(SOLUTIONS, NEURONAL_SOLUTIONS, columns_list, n_col)

