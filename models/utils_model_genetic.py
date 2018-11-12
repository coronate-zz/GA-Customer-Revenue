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

#MODEL= "test"; len_population = 0; len_pc = 0; len_pm= 0 ; N_WORKERS =16



def generate_model_population(columns_list, NEURONAL_SOLUTIONS, n_col, N, max_features):
    POPULATION = dict()
    baseline_all_features = columns_list
    exmodel_features = get_exmodel_features( NEURONAL_SOLUTIONS, n_col )

    for i in range(N):
        POPULATION[i] = dict()

        baseline_features_selected = list()
        baseline_features_chromosome = "0"*len(baseline_all_features)
        n_total_selected_features = random.randint(0, max_features)
        n_baseline_selected_features = random.randint(0, len(baseline_all_features))

        if n_baseline_selected_features > n_total_selected_features:
            n_baseline_selected_features =  n_total_selected_features

        selected_features = random.sample(range(len(baseline_all_features)), n_baseline_selected_features)

        for feature_position in selected_features:
            baseline_features_selected.append(baseline_all_features[feature_position])
            baseline_features_chromosome = list(baseline_features_chromosome)
            baseline_features_chromosome[feature_position] = "1"
            baseline_features_chromosome = "".join(baseline_features_chromosome)

        POPULATION[i]["baseline_features"] =  baseline_features_selected
        POPULATION[i]["baseline_features_chromosome"] = baseline_features_chromosome

        exmodel_features_selected = list()
        if len(exmodel_features) >0: #Not layer N_0
            exmodel_features_chromosome = "0"*len(exmodel_features)
            n_exmodel_selected_features = n_total_selected_features - n_baseline_selected_features
            if n_exmodel_selected_features > len(exmodel_features):
                n_exmodel_selected_features =  len(exmodel_features) -1 
            selected_features = random.sample(range(len(exmodel_features)), n_exmodel_selected_features)
            for feature_position in selected_features:
                exmodel_features_selected.append(exmodel_features[feature_position])
                exmodel_features_chromosome = list(exmodel_features_chromosome)
                exmodel_features_chromosome[feature_position] = "1"
                exmodel_features_chromosome = "".join(exmodel_features_chromosome)
        else:
            exmodel_features_chromosome = ""
            exmodel_features_selected = []

        POPULATION[i]["exmodel_features"] =  exmodel_features_selected
        POPULATION[i]["exmodel_features_chromosome"] = exmodel_features_chromosome
        POPULATION[i]["GENOMA"] = POPULATION[i]["baseline_features_chromosome"] +""+ POPULATION[i]["exmodel_features_chromosome"]
        POPULATION[i]["SCORE"]  = np.nan
    return POPULATION

def solve_genetic_algorithm(N, PC, PM, N_WORKERS, MAX_ITERATIONS, MODEL, NEURONAL_SOLUTIONS, n_col, columns_list, 
    df_kfolded, df_exmodel = None, max_features=1000, round_prediction= False, parallel_execution = False ):
    """
    Solve a Genetic Algorithm with 
        len(genoma)               == GENLONG, 
        POPULATION                == N individuals
        Mutation probability      == PM
        Crossover probability     == PC
        Max number of iterations  == MAX_ITERATIONS
        Cost function             == MODEL
    """

    print("\n\nTEST KFOLDED MEANS: ")
    for fold in df_kfolded.keys():
        print("FOLD", np.mean(df_kfolded[fold]["y"]))

    STILL_CHANGE = True 
    still_change_count = 0 

    if parallel_execution:
        LOCK = multiprocessing.Lock()
        manager_solutions = multiprocessing.Manager()
        SOLUTIONS = manager_solutions.dict()
    else: 
        SOLUTIONS = dict()

    POPULATION_X = generate_model_population(columns_list, NEURONAL_SOLUTIONS, n_col, N, max_features)

    if parallel_execution:
        print("\n\n----------------PARALLEL SOLVE--------------")
        POPULATION_X = parallel_solve(POPULATION_X, SOLUTIONS, LOCK, N_WORKERS, MODEL,
                                      df_kfolded, df_exmodel, "MSE", max_features, round_prediction)
    else:
        print("\n\n--------------SEQUENTIAL SOLVE----------------")
        POPULATION_X = sequential_solve(POPULATION_X, SOLUTIONS, N_WORKERS, MODEL,
                                      df_kfolded, df_exmodel, "MSE", max_features, round_prediction)

    print("\n\n1 -----------------TEST: fenotye in SOLUTIONS acordingly to genoma")
    test_fenotype_chromosome_SOLUTIONS(SOLUTIONS, NEURONAL_SOLUTIONS, columns_list, n_col)

    POPULATION_X = sort_population(POPULATION_X)
    n_iter = 0 
    while (n_iter <= MAX_ITERATIONS) & STILL_CHANGE:

        POPULATION_Y = cross_mutate( POPULATION_X, NEURONAL_SOLUTIONS, columns_list, n_col, N, PC, PM)
        #print("\n\n******************** baseline_features after cross_mutate")
        #test_baseline_xy(POPULATION_Y, POPULATION_X)

        #RESELECT parallel_solve
        if parallel_execution:
            print("\n\n----------------PARALLEL SOLVE--------------\n POPULATION_X length: {} POPULATION_Y length{}".format(len(POPULATION_X.keys()), len(POPULATION_Y.keys())))
            POPULATION_X = parallel_solve(POPULATION_X, SOLUTIONS, LOCK, N_WORKERS, MODEL, 
                           df_kfolded, df_exmodel, "MSE", max_features, round_prediction)
            POPULATION_Y = parallel_solve(POPULATION_Y, SOLUTIONS, LOCK, N_WORKERS, MODEL,
                           df_kfolded, df_exmodel, "MSE", max_features, round_prediction)

        else:
            print("\n\n--------------SEQUENTIAL SOLVE----------------\n POPULATION_X: {} POPULATION_Y {}".format(len(POPULATION_X.keys()), len(POPULATION_Y.keys())))
            POPULATION_X = sequential_solve(POPULATION_X, SOLUTIONS, N_WORKERS, MODEL, 
                           df_kfolded, df_exmodel, "MSE", max_features, round_prediction)
            POPULATION_Y = sequential_solve(POPULATION_Y, SOLUTIONS, N_WORKERS, MODEL,
                           df_kfolded, df_exmodel, "MSE", max_features, round_prediction)
            #print("\n\ntest_baseline_xy 1")
            #test_baseline_xy(POPULATION_X, POPULATION_Y)
        print("\n\n2 -----------------TEST: fenotye in SOLUTIONS acordingly to genoma")
        test_fenotype_chromosome_SOLUTIONS(SOLUTIONS, NEURONAL_SOLUTIONS, columns_list, n_col)
        #print("\n\ntest_baseline_xy 2")
        #test_baseline_xy(POPULATION_X, POPULATION_Y)

        POPULATION_X = sort_population(POPULATION_X)
        POPULATION_Y = sort_population(POPULATION_Y)
        POPULATION_X = select_topn(POPULATION_X, POPULATION_Y, N)

        #print("\n\ntest_baseline_xy 3")
        #test_baseline_xy(POPULATION_X, POPULATION_X)

        equal_individuals, max_score = population_summary(POPULATION_X)

        n_iter += 1
        if equal_individuals >= len(POPULATION_X.keys())*.8:
            still_change_count +=1 
            #print("SAME ** {} ".format(still_change_count))
            if still_change_count >= 10:
                print("\n\n\nGA Solved: \n\tN: {}\n\tPC:{}, \n\tPM: {}\n\tN_WORKERS: {} \n\tMAX_ITERATIONS: {}\n\tMODEL: {}".format( N, PC, PM, N_WORKERS, MAX_ITERATIONS, MODEL ))
                STILL_CHANGE = False

    return POPULATION_X, SOLUTIONS


def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

def test_baseline_xy(POPULATION_X, POPULATION_Y):
    equal_individuals = 0
    for key in POPULATION_Y.keys():
        baseline_features_y =  POPULATION_Y[key]["baseline_features"]
        baseline_features_x =  POPULATION_X[key]["baseline_features"]
        if baseline_features_x == baseline_features_y:
           equal_individuals += 1 
    if equal_individuals == len(POPULATION_X.keys()):
        print("\t WARNING: baseline_features are the same in POPULATION_X and POPULATION_Y")
    print("\t BASELINE: equal_individuals in POPULATION_X and POPULATION_Y: {} out of X{}, Y{} ".format( equal_individuals, len(POPULATION_Y), len(POPULATION_X)))


def test_chromosome_xy(POPULATION_X, POPULATION_Y):
    equal_individuals = 0
    for key in POPULATION_Y.keys():
        baseline_features_y =  POPULATION_Y[key]["baseline_features_chromosome"]
        baseline_features_x =  POPULATION_X[key]["baseline_features_chromosome"]
        if baseline_features_x == baseline_features_y:
           equal_individuals += 1 
    if equal_individuals == len(POPULATION_X.keys()):
        print("\t WARNING: baseline_features are the same in POPULATION_X and POPULATION_Y")
    print("\t CHROMOSOME: equal_individuals in POPULATION_X and POPULATION_Y: {} out of X{}, Y{} ".format( equal_individuals, len(POPULATION_Y), len(POPULATION_X)))



def parallel_solve(POPULATION_X, SOLUTIONS, LOCK, N_WORKERS, MODEL, df_kfolded, df_exmodel, error_type, max_features, round_prediction):
    """
    Each individual in POPULATION_X is assigned to a different process to score it's own
    genoma. Each genoma is scored based on the scoring function and parameters saved individual_score
    MODEL dictionary.
    """
    parallel_execution = True
    s = 0
    POPULATION_SIZE = len(POPULATION_X)
    while s < POPULATION_SIZE-1: 
        #EN esta parte antes de meterlo a procesamiento deberia buscar la solucion...
        started_process = list()
        active_workers = 0

        while active_workers < N_WORKERS:
            if s >= POPULATION_SIZE:
                break
            else:
                if POPULATION_X[s]["GENOMA"] in SOLUTIONS.keys():
                    if MODEL["model_name"] in ["LR"]:
                        #print("Modelo encontrado en ", MODEL["model_name"])
                        continue
                    s+=1
                else:
                    #<------SOLVE
                    #TO DO: tenemos Z started_processes, estos deberian repartire los cores
                    started_process.append(s)
                    s+=1
                    active_workers +=1
        model_name = MODEL["model_name"]
        for s in started_process:
            CORES_PER_SESION = MODEL["params"]["n_workers"]
            time.sleep( random.randint(0,len(started_process)) * .05)
            INDIVIDUAL = copy.deepcopy(POPULATION_X[s])
            POPULATION_X[s]["PROCESS"] = multiprocessing.Process( target= MODEL["function"] , 
                                        args=(INDIVIDUAL, model_name, SOLUTIONS, 
                                        CORES_PER_SESION, LOCK, MODEL, df_kfolded, df_exmodel, error_type,
                                        max_features, round_prediction, parallel_execution, ))

            POPULATION_X[s]["PROCESS"].start()

        #print("STARTED PROCESSES: \n\t", started_process)

        for sp in started_process:  # <---CUANTOS CORES TENGO 
            POPULATION_X[sp]["PROCESS"].join()
            del POPULATION_X[sp]["PROCESS"]

    #save_obj(SOLUTIONS, "ga_solutions")
    #All genoma in SOLUTIONS
    for s in POPULATION_X.keys():
        if  POPULATION_X[s]["GENOMA"] in SOLUTIONS.keys():
            POPULATION_X[s]["SCORE"] = SOLUTIONS[POPULATION_X[s]["GENOMA"]]["score"]
            POPULATION_X[s]["PROB"] = POPULATION_X[s]["SCORE"] / POPULATION_SIZE
        else:
                print("**WARNING: GENOMA not found in SOLUTIONS after scoring")

    return POPULATION_X


def sequential_solve(POPULATION_X, SOLUTIONS, N_WORKERS, MODEL, df_kfolded, df_exmodel, 
                    error_type, max_features, round_prediction):
    """
    Each individual in POPULATION_X is assigned to a different process to score it's own
    genoma. Each genoma is scored based on the scoring function and parameters saved individual_score
    MODEL dictionary.
    """
    s = 0
    model_name = MODEL["model_name"]
    POPULATION_SIZE = len(POPULATION_X)
    for s in POPULATION_X.keys():
        if POPULATION_X[s]["GENOMA"] in SOLUTIONS.keys():
            #print("Modelo encontrado en model_namel: ", MODEL["model_name"])
            continue
        else:
            CORES_PER_SESION = MODEL["params"]["n_workers"]
            INDIVIDUAL = copy.deepcopy(POPULATION_X[s])
            score_model(INDIVIDUAL, model_name, SOLUTIONS, 
                    CORES_PER_SESION, None, MODEL, df_kfolded, df_exmodel, 
                    error_type, max_features, round_prediction, parallel_execution = False)

    for s in POPULATION_X.keys():
        if  POPULATION_X[s]["GENOMA"] in SOLUTIONS.keys():
            POPULATION_X[s]["SCORE"] = SOLUTIONS[POPULATION_X[s]["GENOMA"]]["score"]
            POPULATION_X[s]["PROB"] = POPULATION_X[s]["SCORE"] / POPULATION_SIZE
        else:
            print("**WARNING: GENOMA not found in SOLUTIONS after scoring")

    return POPULATION_X


def score_model(INDIVIDUAL, model_name, SOLUTIONS, CORES_PER_SESION, LOCK, MODEL, df_kfolded, df_exmodel, error_type, max_features, round_prediction = False, parallel_execution = True):
    """
    Scores the model inside a neuron given a particular set of variables and calculates
    the ERROR_TYPES[error_type] for each fold in df_kfolded.
    """
    genoma =  INDIVIDUAL["GENOMA"]
    baseline_features = INDIVIDUAL["baseline_features"]
    exmodel_features  = INDIVIDUAL["exmodel_features"]
    baseline_features_chromosome = INDIVIDUAL["baseline_features_chromosome"]
    exmodel_features_chromosome  = INDIVIDUAL["exmodel_features_chromosome"]
    if df_exmodel:
        total_features = len(baseline_features) + len(exmodel_features) 
    else: 
        total_features = len(baseline_features)
    #print("\n\nTEST SKLEARN_MODELS: {} \n ERROR_TYPES: {} ".format(SKLEARN_MODELS, ERROR_TYPES))
    model = copy.deepcopy(MODEL["model_class"]) #XGBOOST(XGBRegressor, 1)
    total_error = 0
    if total_features > max_features: #or total_features == 0:
        print("WARNING: model not evelauted, number of features bigger than max_features or equal to zero: ", total_features)
        total_error = 1000000000000000000000000000000000000
    else:

        for test_fold in df_kfolded.keys():
            #print("test_fold: ", test_fold)
            if test_fold == "all_data":
                continue
            else:
                X_test_baseline  = df_kfolded[test_fold]["data"].copy()
                X_test_baseline  = X_test_baseline[baseline_features]
                if df_exmodel:
                    X_test_exmodel = df_exmodel[test_fold]
                    X_test_exmodel = X_test_exmodel[exmodel_features]
                    X_test = X_test_baseline.merge(X_test_exmodel, right_index = True, left_index =  True)
                else:
                    X_test = X_test_baseline

                y_test  = df_kfolded[test_fold]["y"].copy()

                X_train_baseline = pd.DataFrame()
                y_train = pd.DataFrame()

                for train_fold in df_kfolded.keys():
                    #print("train_fold: ", train_fold)
                    if train_fold == "all_data" or train_fold == test_fold:
                        continue 
                    else:
                        if len(X_train_baseline)==0:
                            X_train_baseline = df_kfolded[train_fold]["data"]
                            X_train_baseline = X_train_baseline[baseline_features]
                            if df_exmodel:
                                X_train_exmodel  = df_exmodel[train_fold]
                                X_train_exmodel  = X_train_exmodel[exmodel_features]
                                X_train = X_train_baseline.merge(X_train_exmodel, right_index = True, left_index = True)
                            else:
                                X_train = X_train_baseline

                            y_train = df_kfolded[train_fold]["y"].copy()

                        else:
                            X_train_baseline_append = df_kfolded[train_fold]["data"]
                            X_train_baseline_append = X_train_baseline_append[baseline_features]
                            if df_exmodel:
                                X_train_exmodel_append = df_exmodel[train_fold]
                                X_train_exmodel_append = X_train_exmodel_append[exmodel_features]
                                X_train_append = X_train_baseline_append.merge(X_train_exmodel_append, right_index = True, left_index=True)
                            else:
                                X_train_append = X_train_baseline_append

                            X_train = X_train.append(X_train_append)
                            y_train = y_train.append(df_kfolded[train_fold]["y"].copy()) 
                #test_wrongfold_assignation(X_train, X_test)
                model.fit(X_train, y_train, X_test, y_test)
                #time.sleep(.001)
                prediction   = model.predict(X_test)
                prediction[prediction < 0] = 0
                y_test = np.log1p(y_test)

                #print("\n\nPRUEBA prediction: {} \n y_test {}, \n difference: {}".format( prediction[:10], y_test.mean(), np.mean(prediction - y_test)))
                #if round_prediction:
                #    prediction = np.round(prediction)
                error = error_function(y_test, prediction)
                total_error += error
    
        score =  - (total_error/(len(df_kfolded.keys())-1)) - total_features*.00001
        genoma_solutions = dict()
        genoma_solutions["score"] = score
        genoma_solutions["genoma"] = genoma
        genoma_solutions["baseline_features"] = baseline_features
        genoma_solutions["exmodel_features"] = exmodel_features
        genoma_solutions["model"] = model
        genoma_solutions["exmodel_features_chromosome"] =exmodel_features_chromosome
        genoma_solutions["baseline_features_chromosome"] = baseline_features_chromosome

    if parallel_execution:
        #print("\n\n RESULTS: \n\ttotal_error: {} \
        #\n\tlength_error: {}  \n\ttotal_features: {}\
        #\n\tlen baseline_features: {} \n\tbaseline_features: {}\
        #\n\t score{} ".format(- (total_error/(len(df_kfolded.keys())-1)), - total_features*.00001,\
        #total_features, len(baseline_features),baseline_features, score))
        time.sleep( random.randint(0,16) * .05)
        LOCK.acquire()
        SOLUTIONS[genoma] = genoma_solutions
        LOCK.release()

    else:
        #print("\n\n RESULTS: \n\ttotal_error: {} \
        #\n\tlength_error: {}  \n\ttotal_features: {}\
        #\n\tlen baseline_features: {} \n\tbaseline_features: {}\
        #\n\t score{} ".format(- (total_error/(len(df_kfolded.keys())-1)), - total_features*.00001,\
        #total_features, len(baseline_features),baseline_features, score))

        SOLUTIONS[genoma]= genoma_solutions




def test_wrongfold_df_kfolded(df_kfolded):
    flag = False
    for fold in df_kfolded.keys():
        data_fold = df_kfolded[fold]["data"]
        if fold == "all_data":
            duplicates_alldata = np.sum(df_kfolded[fold]["data"].duplicated())
        else:
            other_data = pd.DataFrame()
            for fold2 in df_kfolded.keys():
                if fold2 != fold and fold2 != "all_data":
                    if len(other_data) ==0:
                        other_data = df_kfolded[fold2]["data"]
                    else :
                        other_data = other_data.append(df_kfolded[fold2]["data"])
                else:
                    continue
            duplicates_otherdata =  np.sum(other_data.duplicated())
            duplicates_inside_fold = np.sum(data_fold.duplicated())
            all_data = data_fold.append(other_data)
            duplicates_alldata =  np.sum(all_data.duplicated())
            if duplicates_alldata > 0 :
                print("\n\nWARNING: Duplicates found at fold {}: \n\tother_data:{}  inside_fold: {} all_data pct{}".format(fold, duplicates_otherdata, duplicates_inside_fold, duplicates_alldata/len(all_data) ))
                flag = True
    if not flag:
        print("\t **test_wrongfold_df_kfolded passed !")

def test_wrongfold_assignation(X_train, X_test):
    X = X_train.append(X_test)
    try:
        duplicates= np.sum(X.duplicated())
        if duplicates >0:
            print("WARNING: duplicated values in test set -  ", duplicates)
    except Exception as e:
        print(e)


def error_function(y_test, prediction):
    difference =  y_test -prediction
    difference_abs =  np.abs(difference)
    error  = difference_abs.mean()
    return error



def score_genetics(genoma, SOLUTIONS, CORES_PER_SESION, LOCK, MODEL ):
    """
    This scoring fucntion is use to test how good is the configuration of a 
    genetic algorithm to solve a problem of GENLONG 
    """
    end_population = MODEL["params"]["len_population"]
    end_pc         = end_population + MODEL["params"]["len_pc"]
    end_pm         = end_pc         + MODEL["params"]["len_pm"]

    POPULATION_SIZE  = int(genoma[:end_population], 2) +1
    PC               = 1/(int(genoma[end_population: end_pc], 2) +.0001)
    PM               = 1/(int(genoma[end_pc: end_pm], 2) +.0001)

    MAX_ITERATIONS = MODEL["params"]["max_iterations"]
    GENLONG        = MODEL["params"]["len_genoma"] + 1
    N_WORKERS      = MODEL["params"]["n_workers"]

    print("\n\n\nEXECUTE SCORE GENETICS: \n\POPULATION_SIZE: {}\n\tpc: {} \n\tPM: {}".format(POPULATION_SIZE, PC, PM ))

    time_start = time.time()
    test ={"model_type": "test",  "function": score_test, "params": { "n_workers": 4}}
    POPULATION_X, SOLUTIONS_TEST = solve_genetic_algorithm(GENLONG, POPULATION_SIZE, PC, PM,  N_WORKERS, MAX_ITERATIONS, test)
    time_end = time.time()
    x, max_score = population_summary(POPULATION_X)


    TOTAL_TIME = time_end - time_start
    SCORE = GENLONG- max_score
    final_score = -(SCORE) -(.1 *TOTAL_TIME)
    print("\t\tTIME: {}  MAX_SCORE: {} FINAL_SCORE: {}".format(TOTAL_TIME, max_score, final_score))

    LOCK.acquire()
    SOLUTIONS[genoma]["score"]  = final_score
    LOCK.release()




def score_test(genoma, SOLUTIONS,  CORES_PER_SESION, LOCK, MODEL):
    #print("\n\n\n MODEL TYPE: {} , \n\tGENOMA: {}, ".format( MODEL["model_type"], genoma, SOLUTIONS))
    suma = 0 
    for i in genoma:
        suma += int(i)
    time.sleep(.1)
    LOCK.acquire()
    SOLUTIONS[genoma]["score"] = suma
    LOCK.release()


def save_solutions(genoma, new_score, SOLUTIONS):
    SOLUTIONS[genoma]["score"] = new_score


def get_exmodel_features(NEURONAL_SOLUTIONS, eval_layer):
    """
    Append all exmodel output features names to call them from the df_exmodel dataset.
    """
    exmodel_features =list()
    for n_col in NEURONAL_SOLUTIONS.keys():
        for m_row in NEURONAL_SOLUTIONS[n_col].keys():
            if int(eval_layer.replace("N_", "")) > int(n_col.replace("N_", "")):
                #A model can use an other neuron output iff the layer of the model is lower
                #than the neurons layer, i.e it can only used output from already trained 
                #models.
                exmodel_features.append(NEURONAL_SOLUTIONS[n_col][m_row]["name"])
    return exmodel_features


def sort_population(POPULATION_X):
    POPULATION_X_copy = copy.deepcopy(POPULATION_X)
    POPULATION_sorted = sorted(POPULATION_X_copy.items(), key=lambda x: x[1]["SCORE"], reverse = True)
    POPULATION_NEW = dict()
    result = list()
    cont = 0
    for i in POPULATION_sorted:
        POPULATION_NEW[cont] = i[1]
        cont += 1
    #test_baseline_xy(POPULATION_X, POPULATION_NEW)
    return POPULATION_NEW

def population_summary(POPULATION_X):
    POPULATION_X_copy = POPULATION_X.copy()
    suma = 0
    min_score =  100000000000000000000000
    max_score = -100000000000000000000000
    lista_genomas = list()
    baseline_features = []
    best_baseline_features = []
    best_exmodel_features = []
    for key in POPULATION_X_copy.keys():
        individual_score =  POPULATION_X_copy[key]["SCORE"]
        lista_genomas.append(POPULATION_X[key]["GENOMA"])
        suma += individual_score
        if max_score < individual_score:
            max_score = individual_score
            best_baseline_features = POPULATION_X_copy[key]["baseline_features_chromosome"]
            best_exmodel_features  = POPULATION_X_copy[key]["exmodel_features_chromosome"]
        if min_score > individual_score:
            min_score = individual_score
    promedio = suma/len(POPULATION_X_copy.keys())
    equal_individuals = len(lista_genomas) - len(set(lista_genomas))
    print("\n\nTEST population_summary: \
        \n\tTOTAL SCORE: {} \n\tMEAN SCORE: {} \
        \n\t MAX_SCORE: {} \n\tMIN_SCORE: {} \
        \n\t BASELINE FEAT: {} \n\t EXMODEL FEAT: {}".format(suma, promedio, max_score,\
         min_score, best_baseline_features, best_exmodel_features))

    return equal_individuals, max_score


def select_topn( POPULATION_X, POPULATION_Y, N):
    POPULATION_Y_copy =  copy.deepcopy(POPULATION_Y)
    POPULATION_X_copy = copy.deepcopy(POPULATION_X)
    for key in POPULATION_X_copy.keys():
        new_key = key  + N
        POPULATION_Y_copy[new_key] = POPULATION_X_copy[key]

    POPULATION_Y_copy = sort_population(POPULATION_Y_copy)
    POPULATION_NEW =dict()

    for key in range(N):
        POPULATION_NEW[key] = POPULATION_Y_copy[key]

    return POPULATION_NEW


def cross_mutate( POPULATION_X, NEURONAL_SOLUTIONS, columns_list, n_col, N, PC, PM):
    POPULATION_Y = copy.deepcopy(POPULATION_X)
    #pprint(POPULATION_X)
    for j in range(int(N/2)):
        pc = random.uniform(1,0)
        GENLONG =  len(POPULATION_X[0]["GENOMA"])
        #CROSSOVER
        if pc < PC:
            best = POPULATION_Y[j]["GENOMA"]
            worst = POPULATION_Y[N -j-1]["GENOMA"]
            startBest =  best
            startWorst = worst

            genoma_crossover = random.randint(0, GENLONG)
            genoma_crossover_final = int(genoma_crossover + np.round(GENLONG/2))
            if genoma_crossover_final > GENLONG:
                # To perform a roulette  cross over we need to fill the
                extra_genes = genoma_crossover_final - GENLONG 
                genoma_crossover_final =  GENLONG

            else: 
                extra_genes = 0

            best_partA  = best[:extra_genes]
            worst_partA = worst[:extra_genes]

            best_partB  = best[extra_genes:genoma_crossover]
            worst_partB = worst[extra_genes:genoma_crossover]

            best_partC  = best[genoma_crossover:genoma_crossover_final]
            worst_partC = worst[genoma_crossover: genoma_crossover_final]

            best_partD  = best[genoma_crossover_final:GENLONG]
            worst_partD = worst[genoma_crossover_final: GENLONG]

            #print("TEST CROSSOVER:   {} -> {} -> {} -> {} \npart A {} \npartB {} \npart C{} \npartD {}".format(extra_genes, genoma_crossover, genoma_crossover_final, GENLONG, best_partA, best_partB, best_partC, best_partD) ) 
            new_best    = worst_partA + best_partB + worst_partC + best_partD
            new_worst   = best_partA + worst_partB + best_partC + worst_partD

            endBest = new_best
            endWorst =  new_worst

            POPULATION_Y[j]["GENOMA"]     = new_best
            POPULATION_Y[N-j-1]["GENOMA"] = new_worst

            #print("\n\nCrossover Performed on individual {}-{}: \n\t StartBest: {} \n\t NewBest: {} \n\t StartWorst: {} \n\t NewWorst: {}".format(j,genoma_crossover, startBest, endBest, startWorst, endWorst))

    for j in range(N):
        #MUTATION

        pm = random.uniform(1,0)
        mutation_gen = random.randint(0, GENLONG-1)
        mutated_genoma = POPULATION_Y[j]["GENOMA"]

        if pm < PM:
            #PERFORM MUTATION TO DO: needs to make the mutation for each gen not for single mutation_gen

            start = mutated_genoma
            if mutated_genoma[mutation_gen] =="1":
                mutated_genoma = list(mutated_genoma)
                mutated_genoma[mutation_gen] = "0"
                mutated_genoma = "".join(mutated_genoma)
            else:
                mutated_genoma = list(mutated_genoma)
                mutated_genoma[mutation_gen] = "1"
                mutated_genoma = "".join(mutated_genoma)

        POPULATION_Y[j]["GENOMA"] = mutated_genoma
        POPULATION_Y[j]["baseline_features_chromosome"] = mutated_genoma[:len(columns_list)]
        POPULATION_Y[j]["exmodel_features_chromosome"]  = mutated_genoma[len(columns_list):]

    #print("\n\n0 ******************** deben ser iguales\nTEST POPULATION diff POPULATION_Y")
    #test_baseline_xy(POPULATION_Y, POPULATION_X) #Prueba de que el genotipo ha sido efectivo
    #print("\n\n1 -------------------- deben ser diferentes se ha cambiado solo los cromosomas\nTEST POPULATION diff POPULATION_Y")
    #test_chromosome_xy(POPULATION_Y, POPULATION_X)
    #print("\n\n\n -------------------Here most mark Error")
    #test_fenotype_chromosome_POPULATION(POPULATION_Y, NEURONAL_SOLUTIONS, df_kfolded, n_col)
    POPULATION_Y = change_fenotype_using_genoma(POPULATION_Y, NEURONAL_SOLUTIONS, n_col, columns_list)
    print("\n\n**Second test in test_fenotype_chromosome_POPULATION")
    test_fenotype_chromosome_POPULATION(POPULATION_Y, NEURONAL_SOLUTIONS, columns_list, n_col)
    #print("\n\n2 ******************** deben cambiar \nTEST POPULATION diff POPULATION_Y")
    #test_baseline_xy(POPULATION_Y, POPULATION_X) #Prueba de que el genotipo ha sido efectivo

    return POPULATION_Y





def  change_fenotype_using_genoma(POPULATION, NEURONAL_SOLUTIONS, n_col, columns_list):
    """ After modifying the genoma of an individual it's necessary to change the fenotype
    acordingly to the genotype. In other words this fucntion change the "shape" or characteristics
    of the individual (columns used in the model fitting)  ti fit the model.
    """
    for individual in POPULATION.keys():
        del POPULATION[individual]["exmodel_features"]
        del POPULATION[individual]["baseline_features"]

    for individual in POPULATION.keys():
        cont = 0
        baseline_all_features = columns_list.copy()
        exmodel_features = get_exmodel_features( NEURONAL_SOLUTIONS, n_col )

        baseline_features_selected = list()
        exmodel_features_selected = list()
        for chromosome in POPULATION[individual]["baseline_features_chromosome"]:
            if int(chromosome) == 1:
                feature_inbit = baseline_all_features[cont]
                baseline_features_selected.append(feature_inbit)
            cont += 1 
        cont = 0
        for chromosome in POPULATION[individual]["exmodel_features_chromosome"]:
            if int(chromosome) == 1:
                feature_inbit = exmodel_features[cont]
                exmodel_features_selected.append(exmodel_features[cont])
            cont += 1 

        POPULATION[individual]["exmodel_features"] = exmodel_features_selected
        POPULATION[individual]["baseline_features"] = baseline_features_selected
    print("\n\n\n -------------------Here most be ok")
    test_fenotype_chromosome_POPULATION(POPULATION, NEURONAL_SOLUTIONS, columns_list, n_col)

    return  POPULATION




def  test_fenotype_chromosome_POPULATION(POPULATION, NEURONAL_SOLUTIONS, columns_list, n_col):
    """ For each individual in POPULATION, compares that the fenotye(selected_features) correspond 
    to the sequence of the bits in the chromosome.
    """
    print("Performing test_fenotype_chromosome on POPULATION")
    for individual in POPULATION.keys():
        baseline_features  = columns_list.copy()
        exmodel_features = get_exmodel_features( NEURONAL_SOLUTIONS, n_col )
        cont = 0
        for chromosome in POPULATION[individual]["baseline_features_chromosome"]:
            feature_inbit  = baseline_features[cont]
            if int(chromosome) == 1:
                #feature included
                if feature_inbit not in  POPULATION[individual]["baseline_features"]:
                    raise ValueError("ERROR:test_fenotype_chromosome_POPULATION\n\t feature not in baseline_features while {} chromosome is 1: {}".format(cont,feature_inbit))
            else: 
                if feature_inbit in POPULATION[individual]["baseline_features"]:
                    raise ValueError("ERROR:test_fenotype_chromosome_POPULATION\n\t  feature in baseline_features while {} chromosome is 1: {}".format(cont,feature_inbit))
            cont +=1 
        if POPULATION[individual]["baseline_features_chromosome"] != POPULATION[individual]["GENOMA"][:len(columns_list)]:
            raise ValueError("ERROR: test_fenotype_chromosome_POPULATION \n\t baseline_features_chromosome different from genoma[:x]" )

        cont = 0
        for chromosome in POPULATION[individual]["exmodel_features_chromosome"]:
            feature_inbit  = exmodel_features[cont]
            if int(chromosome) == 1:
                #feature included
                if feature_inbit not in  POPULATION[individual]["exmodel_features"]:
                    raise ValueError("ERROR: feature not in exmodel_features while {} chromosome is 1: {}".format(cont,feature_inbit))
            else: 
                if feature_inbit in POPULATION[individual]["exmodel_features"]:
                    raise ValueError("ERROR: feature in exmodel_features while {} chromosome is 0: {} ".format(cont, feature_inbit))
            cont +=1 
        if POPULATION[individual]["exmodel_features_chromosome"] != POPULATION[individual]["GENOMA"][len(columns_list):]:
            raise ValueError("ERROR: test_fenotype_chromosome_POPULATION \n\t exmodel_features_chromosome different from genoma[:x]" )


    print("\n\t**TEST PASSED: test_fenotype_chromosome_POPULATION")


def  test_fenotype_chromosome_SOLUTIONS(SOLUTIONS, NEURONAL_SOLUTIONS, columns_list, n_col):
    """ For each individual in POPULATION, compares that the fenotye(selected_features) correspond 
    to the sequence of the bits in the chromosome.

    This tests change_fenotype_using_genoma
    """
    print("Performing test_fenotype_chromosome on SOLUTIONS")
    for individual in SOLUTIONS.keys():
        baseline_features  = columns_list.copy()
        exmodel_features = get_exmodel_features( NEURONAL_SOLUTIONS, n_col )
        cont = 0
        for chromosome in SOLUTIONS[individual]["baseline_features_chromosome"]:
            feature_inbit  = baseline_features[cont]
            if int(chromosome) == 1:
                #feature included
                if feature_inbit not in  SOLUTIONS[individual]["baseline_features"]:
                    raise ValueError("ERROR: feature not in baseline_features while {} chromosome is 1: {}".format(cont,feature_inbit))
            else: 
                if feature_inbit in SOLUTIONS[individual]["baseline_features"]:
                    raise ValueError("ERROR: feature in baseline_features while {} chromosome is 1: {}".format(cont,feature_inbit))
            cont +=1 
        cont = 0
        for chromosome in SOLUTIONS[individual]["exmodel_features_chromosome"]:
            feature_inbit  = exmodel_features[cont]
            if int(chromosome) == 1:
                #feature included
                if feature_inbit not in  SOLUTIONS[individual]["exmodel_features"]:
                    raise ValueError("ERROR: feature not in exmodel_features while {} chromosome is 1: {}".format(cont,feature_inbit))
            else: 
                if feature_inbit in SOLUTIONS[individual]["exmodel_features"]:
                    raise ValueError("ERROR: feature in exmodel_features while {} chromosome is 0: {} ".format(cont, feature_inbit))
            cont +=1 
    print("\n\t**TEST PASSED: test_fenotype_chromosome_SOLUTIONS")


def report_genetic_results(genoma, MODEL):
    end_population = MODEL["params"]["len_population"]
    end_pc         = end_population + MODEL["params"]["len_pc"]
    end_pm         = end_pc         + MODEL["params"]["len_pm"]

    POPULATION_SIZE  = int(genoma[:end_population], 2) +1
    PC               = 1/(int(genoma[end_population: end_pc], 2) +.01)
    PM               = 1/(int(genoma[end_pc: end_pm], 2) +.01)

    MAX_ITERATIONS = MODEL["params"]["max_iterations"]
    GENLONG        = MODEL["params"]["len_genoma"] + 1
    N_WORKERS      = MODEL["params"]["n_workers"]
    print("\n\n ** BEST GA **: \n\tgenlong: {}\n\tpc: {} \n\tPM: {}".format(POPULATION_SIZE, PC, PM ))



def test_equal_squares(SOLUTIONS):
    equal_scores_keys = dict()
    for key in SOLUTIONS.keys():
        equal_scores_keys[key] = list()
        base_score = SOLUTIONS[key]["score"]
        equal_scores = 0
        for key2 in SOLUTIONS.keys():
            if key != key2:
                compare_score =  SOLUTIONS[key2]["score"]
                if base_score == compare_score:
                    equal_scores += 1
                    equal_scores_keys[key].append(key2)

        if equal_scores >0:
            print("\tTest equal_scores: {} repeated scores for SOLUTIONS[{}]".format( equal_scores, key))
        else:
            del equal_scores_keys[key] 
    return equal_scores_keys

def test_equal_squares_likelyness(equal_scores_keys, SOLUTIONS):
    warning_flag = False
    for s1 in equal_scores_keys.keys():
        for s2 in equal_scores_keys[s1]:
            equal_count = 0
            len_genoma = len(SOLUTIONS[s1]["genoma"])
            for i in range(len_genoma-1):
                e1 = SOLUTIONS[s1]["genoma"][i]
                e2 = SOLUTIONS[s1]["genoma"][i]
                if e1 == e2:
                    equal_count += 1
            if (equal_count/len_genoma) < .95: 
                print("\n\n{} \n\tand \n{} \n\t are {} of {} alike pct: {} ".format(s1,s2, equal_count, len_genoma, equal_count/len_genoma))
                warning_flag = True
            print("likelyness: \n\ts1{}  \n\ts2{} \n\t pct{}".format(s1, s2, equal_count/len_genoma))
    if warning_flag:
        print("WARNING: Some of the feature configurations are reporting the same value.")
    else:
        print("\t** test_equal_squares_likelyness passed!")

