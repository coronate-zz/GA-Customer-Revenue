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
df = pd.read_csv("../input/df_train_prepared.csv" )
df = df.reset_index()
df =  df.drop_duplicates()
df = df.replace(np.nan, 0)
df = df.sample(10000)
del df["date"]
df["fullVisitorId"] = df.fullVisitorId.astype(float)


df_kfolded = utils_exomodel.transform_KFold_groups(df, "totals.transactionRevenue", 5   , "fullVisitorId")#, "fullVisitorId")

M= 4     
N= 4

neuronal_system, chromosome = utils_exomodel.build_neuronal_system(M, N, models_list)
NEURONAL_SOLUTIONS          = utils_exomodel.decode_neuronal_system(neuronal_system, df, SKLEARN_MODELS)
utils_exomodel.get_intramodel_chromosomes(NEURONAL_SOLUTIONS)
#utils_exomodel.simulate_output(NEURONAL_SOLUTIONS, N, df_kfolded)
#df_exmodel = utils_exomodel.get_models_output(NEURONAL_SOLUTIONS, df_kfolded) #runs everytime a layer is finished.



N = 15
PC =.13   #Crossover probability
PM =.13 #Mutation probability
MAX_ITERATIONS = 8

N_WORKERS = 5
max_features = 1000

n_col = "N_0"
columns_list = list(df_kfolded["all_data"]["data"].columns)

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
                            MAX_ITERATIONS, MODEL, NEURONAL_SOLUTIONS, n_col, columns_list, df_kfolded,
                            df_exmodel = None , max_features = max_features, round_prediction= True,
                            parallel_execution = False)
t2 = time.time()

#equal_scores_keys = utils_model_genetic.test_equal_squares(SOLUTIONS)
#utils_model_genetic.test_equal_squares_likelyness(equal_scores_keys, SOLUTIONS)
print("\n\nTIME: {} {}  == {} ".format(t1,t2, t2-t1))

selected_features = pd.DataFrame()
selected_features["selected_features"] = POPULATION_X[0]["baseline_features"]
selected_features.to_csv("../input/selected_features_persession.csv", index = False)

utils_model_genetic.test_fenotype_chromosome_SOLUTIONS(SOLUTIONS, NEURONAL_SOLUTIONS, columns_list, n_col)


for individual in SOLUTIONS.keys(): 
    print("\n\n-------------NEW SET----------------------------")
    for k in SOLUTIONS.keys():
        if SOLUTIONS[individual]["genoma"] != SOLUTIONS[k]["genoma"] and SOLUTIONS[k]["baseline_features_chromosome"] == SOLUTIONS[individual]["baseline_features_chromosome"]:
            raise ValueError("individual in SOLUTIONS have duplicated fenotype")
cont= 0

for individual in SOLUTIONS.keys(): 
    print("\n\n-------------NEW SET----------------------------")
    for k in SOLUTIONS.keys():  
        if SOLUTIONS[individual]["baseline_features"] == SOLUTIONS[k]["baseline_features"] and SOLUTIONS[k]["genoma"] != SOLUTIONS[individual]["genoma"]:
            cont +=1
            #raise ValueError("individual in SOLUTIONS have duplicated fenotype")
print(cont)


for individual in SOLUTIONS.keys(): 
    print("\n\n-------------NEW SET----------------------------\n", individual)
    for k in SOLUTIONS.keys():
        if SOLUTIONS[individual]["baseline_features"] == SOLUTIONS[k]["baseline_features"] and SOLUTIONS[k]["genoma"] != SOLUTIONS[individual]["genoma"]:
            print(SOLUTIONS[k]["baseline_features_chromosome"] == SOLUTIONS[individual]["baseline_features_chromosome"])
            print(k)
            print(SOLUTIONS[individual]["score"])
            print(SOLUTIONS[k]["score"])

            #raise ValueError("individual in SOLUTIONS have duplicated fenotype")

