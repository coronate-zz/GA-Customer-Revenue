import pandas as pd
import numpy as np
import random
from xgboost import XGBRegressor
from xgboost import plot_importance
import pickle
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error 
import importlib 
from sklearn import linear_model
from sklearn.model_selection import GroupKFold


def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)


def get_exochromoesome(df, models):
    """
    transforms a neuronal system into a code that represents the models used in each neuron.
    This code is used as a chromosome in the genetic algorithm.
    """
    bits_per_model = int(np.ceil(np.log2(len(models))))
    chromosome = ""
    for n_col in df.columns:
        for m_row in df.index:
            model = df.loc[m_row, n_col]
            model_position = models.index(model)
            chromosome = chromosome + "" + np.binary_repr(model_position, width= bits_per_model )
    return chromosome


def build_neuronal_system(M, N, models):
    """
    Creates a neuranal system represented by a dictionary of M elements each one with N entries.
    In each system a random model is selected for training and score. Models in upper layers of
    the neuranl system get information from models in lower levels.
    """
    n_columns =  list()
    for i in range(N):
        n_name = "N_" +str(i)
        n_columns.append(n_name)
    m_rows =  list()
    for j in range(M):
        m_name = "M_"+str(j)
        m_rows.append(m_name)

    neuronal_system = pd.DataFrame(columns =  n_columns, index = m_rows)
    for n_col in n_columns:
        for m_row in m_rows:
            neuronal_system.loc[m_row, n_col] = random.choice(models)

    chromosome = get_exochromoesome(neuronal_system, models)

    return neuronal_system, chromosome


def decode_neuronal_system(neuronal_system, df, SKLEARN_MODELS):
    """
    This funtion transforms the neuronal_system into a set of models that must be trained 
    in order to score the neuronal_system. NEURONAL_SOLUTIONS represent the neuronal network
        **Note: NEURONAL_SOLUTIONS is different from neuronal_system. The first one reports the score
        while the second one report the charcateristics of the mdoel to be trained.

    NEURONAL_SOLUTIONS[n_col][m_row] = indicates the characteristics of a particular neuron 
    at model_position n_col, m_row.
        Inside each neuron we find:
            * model_name: XGBoost, LR, RandomForest ...
            * model: sklearn object with the model for fit.
            * name: model_name + neuron position. This feature is usefull to identify 
                    the output of the neuron as imput of other neuron
            * baseline_features: The features inside the initial dataset. baseline_features doesn't
                    take into account the output of other neurons.
            * baseline_features_chromosome: The binary representation of the baseline_features.
                    Each fature is represented by one bit. If the model in the corresponded neuron
                    is choosen to be trained with baseline_fetaure X, then the position at X
                    for the baseline_features_chromosome must be equal to 1 (0 if it's not included)
            * exmodel: A dictionary that contains features from other models.
                    There is a entry for each layer in the model:
                          1 2 3
                         [][][][]  models in layer 1, con use output features from layer 0
                         [][][][]  models in layer 2, can use output features from layer 0,1 
                         [][][][]  ...
                         [][][][]  models in layer 3 can use output features from layer 0,1,2
                    * exmodel[layer][features]: For a particular layer, what variables will
                                  be used in the input neuron-model.
                                  **if layer of neuron < layer, non variable is included.
                    * exmodel[layer][chromosome]: representation of the features used
                                  in the neuron model from layer = layerX. 
                                  1 if output of layerX, neuron 1 included
                                  01 if output of layerX, neuron 2 included
                                  ...
                                  0001 if output of layerX, neuron 4 included.
    """
    NEURONAL_SOLUTIONS = dict()
    for i in range(len(neuronal_system.columns)):
        n_col = neuronal_system.columns[i]
        NEURONAL_SOLUTIONS[n_col] = dict()
        for j in range(len(neuronal_system.index)):
            m_row = neuronal_system.index[j]

            model = neuronal_system.loc[m_row, n_col]
            NEURONAL_SOLUTIONS[n_col][m_row] = dict()
            NEURONAL_SOLUTIONS[n_col][m_row]["model_name"] = model
            NEURONAL_SOLUTIONS[n_col][m_row]["model"] = SKLEARN_MODELS[model]
            NEURONAL_SOLUTIONS[n_col][m_row]["name"] = model + "_" + str(n_col) + "_" + str(m_row)

            #For each model we want to test the features that should be included in the model. 
            features_number = len(df.columns)
            baseline_features = list()
            baseline_features_chromosome = ""
            for f in range(features_number):
                feature = df.columns[f]
                isincluded = random.choice([True,False])
                if isincluded:
                    baseline_features.append(feature)
                    baseline_features_chromosome += "1"
                else:
                    baseline_features_chromosome += "0"

            NEURONAL_SOLUTIONS[n_col][m_row]["baseline_features"] =  baseline_features
            NEURONAL_SOLUTIONS[n_col][m_row]["baseline_features_chromosome"] =  baseline_features_chromosome
            NEURONAL_SOLUTIONS[n_col][m_row]["exmodel"] =  dict()
            

            #Include exmodel features for layers>0 
            for layer in neuronal_system.columns:
                NEURONAL_SOLUTIONS[n_col][m_row]["exmodel"][layer] = dict()
                if i <= int(layer.replace("N_", "")) or i ==0:
                    NEURONAL_SOLUTIONS[n_col][m_row]["exmodel"][layer]["features"] = []
                    NEURONAL_SOLUTIONS[n_col][m_row]["exmodel"][layer]["chromosome"] = ""
                else:

                    exmodel = list()
                    exmodel_chromosome = ""
                    for j_2 in range(len(neuronal_system.index)):
                        m_row_asfeature = neuronal_system.index[j_2]
                        feature = NEURONAL_SOLUTIONS[layer][m_row_asfeature]["name"]
                        isincluded = random.choice([True,False])

                        if isincluded and NEURONAL_SOLUTIONS[layer][m_row_asfeature]["model_name"] != "":
                            exmodel.append(feature)
                            exmodel_chromosome += "1"
                        else:
                            exmodel_chromosome += "0"

                    NEURONAL_SOLUTIONS[n_col][m_row]["exmodel"][layer]["features"]   = exmodel
                    NEURONAL_SOLUTIONS[n_col][m_row]["exmodel"][layer]["chromosome"] = exmodel_chromosome

    return NEURONAL_SOLUTIONS


def decode_exochromosome(M, N , chromosome, models):
    bits_per_model = int(np.ceil(np.log2(len(models))))

    n_columns =  list()
    for i in range(N):
        n_name = "N_" +str(i)
        n_columns.append(n_name)
    m_rows =  list()
    for j in range(M):
        m_name = "M_"+str(j)
        m_rows.append(m_name)

    neuronal_system = pd.DataFrame(columns =  n_columns, index = m_rows)
    for i in range(len(n_columns)):
        n_col =  n_columns[i]
        for j in range(len(m_rows)):
            m_row = m_rows[j]
            neuronal_system.loc[m_row, n_col] = chromosome[(i*len(m_rows)*bits_per_model)+ (j*bits_per_model) : (i*len(m_rows)*bits_per_model)+ (j*bits_per_model) +bits_per_model]

    return chromosome



def get_intramodel_chromosomes(NEURONAL_SOLUTIONS):
    """
    return a chromosome that represents the features that will be used to fit the 
    model inside a neuron.
    baseline_features   +   layer0_feature   + layer1_features + ... + layern_features
    [][][][][][][][]        [][][][][][][]     [][][][][][][][]        [][][][][][][]
                            
                            1 bit per neuron    1 bit per neuron
                            in layer 0.         in layer 1.

     """
    for n_col in NEURONAL_SOLUTIONS.keys():
        for m_row in NEURONAL_SOLUTIONS[n_col].keys():
            intramodel_chromosome = ""
            intramodel_chromosome += NEURONAL_SOLUTIONS[n_col][m_row]["baseline_features_chromosome"]
            for i in range(len(NEURONAL_SOLUTIONS[n_col][m_row]["exmodel"].keys())):
                layer = "N_" +  str(i)
                intramodel_chromosome += NEURONAL_SOLUTIONS[n_col][m_row]["exmodel"][layer]["chromosome"]
            NEURONAL_SOLUTIONS[n_col][m_row]["intramodel_chromosome"] = intramodel_chromosome


def get_intramodel_features(NEURONAL_SOLUTIONS, LAYER):
    """
    This function use the exmodel_chromosome of NEURON Z to return output of the
    NEURON Y model  as an input for NEURON Z.
    This function must be applied in order beacause it requieres that the output of NEURON Y
    is ready to be used by NEURON Z.

    in a NEURONAL_SOLUTIONS system of 4X4:
                            layer0       layer1     layer2   layer3
    exmodel_chromosome: [1][0][0][1] [0][0][1][1] [][][][] [][][][]

    In this exmple NEURON Z needs the output neuron0 and neuron3 of layer1
                            needs the output of neuron2 and neuron3 of layer2
                            don't need any input from layer2 and layer3.


    1. Make a data DataFrame with the output of all the neuron models for layer<LAYER
    2. Use the exmodel as a index for the DataFrame in 1.
    3. Merge 3 with baseline_features.

    """

    all_exmodel_features = pd.DataFrame()
    for n_col in range(NEURONAL_SOLUTIONS.keys()):
        if n_col < int(LAYER.replace("N_", "")):
            name = NEURONAL_SOLUTIONS[n_col][m_row]["name"]
            all_exmodel_features[name] = NEURONAL_SOLUTIONS[n_col][m_row]["output"]

    for m_row in NEURONAL_SOLUTIONS[LAYER].keys():
        exmodel_features = list()
        for layer in NEURONAL_SOLUTIONS[LAYER][m_row]["exmodel"].keys():
            exmodel_features.append(NEURONAL_SOLUTIONS[LAYER][m_row]["exmodel"][layer]["features"])

        NEURONAL_SOLUTIONS[LAYER][m_row]["exmodel_features"] = exmodel_features
        NEURONAL_SOLUTIONS[LAYER][m_row]["all_features"] = exmodel_features.append(NEURONAL_SOLUTIONS[LAYER][m_row]["baseline_features"])
        NEURONAL_SOLUTIONS[LAYER][m_row]["exmodel_DataFrame"] = all_exmodel_features[NEURONAL_SOLUTIONS[LAYER][m_row]["exmodel_features"]]
        
        NEURONAL_SOLUTIONS[LAYER][m_row]["all_DataFrame"] = df[NEURONAL_SOLUTIONS[LAYER][m_row]["baseline_features"]]\
                                                            .merge(NEURONAL_SOLUTIONS[LAYER][m_row]["exmodel_DataFrame"], \
                                                            right_index =True, left_index = True) 

    return all_exmodel_features

def simulate_output(NEURONAL_SOLUTIONS, layer, df_kfolded):
    """
    Fills NEURONAL_SOLUTIONS[n_col][m_row]["output"]
    With a simulated output of len(df).
    """
    for n_col in NEURONAL_SOLUTIONS.keys():
        if int(n_col.replace("N_", "")) < layer:
            for m_row in NEURONAL_SOLUTIONS[n_col].keys():
                NEURONAL_SOLUTIONS[n_col][m_row]["output"] = dict()
                NEURONAL_SOLUTIONS[n_col][m_row]["output"]["all_data"] = list()
                for fold in df_kfolded.keys():
                    NEURONAL_SOLUTIONS[n_col][m_row]["output"][fold] = np.random.rand( len(df_kfolded[fold]["data"]))
                    if fold != "all_data":
                        NEURONAL_SOLUTIONS[n_col][m_row]["output"]["all_data"] = np.append(NEURONAL_SOLUTIONS[n_col][m_row]["output"]["all_data"], NEURONAL_SOLUTIONS[n_col][m_row]["output"][fold])


def get_models_output(NEURONAL_SOLUTIONS, df_kfolded):
    """
    For LAYER N > 0, some models may need the predictions of previous models.
    this function returnsa DataFreme with the output of all the previous models runned.
    df_exmodel contains an entry for every fold.
    df_exmodel[fold] = DataFrame cols = Nn_Mm_modelname for all n, m. 
    df_exmodel[fold] = NEURONAL_SOLUTIONS[n_col][m_row]["output][fold] for all n_col, m_row

    """
    df_exmodel = dict()
    for fold in df_kfolded.keys():
        df_exmodel[fold] = pd.DataFrame()
        for n_col in NEURONAL_SOLUTIONS.keys():
            for m_row in NEURONAL_SOLUTIONS[n_col].keys():
                name    = NEURONAL_SOLUTIONS[n_col][m_row]["name"]
                index   = df_kfolded[fold]["index"]
                output  = NEURONAL_SOLUTIONS[n_col][m_row]["output"][fold]
                df_fold = pd.DataFrame( data = output, index =  index, columns = [name])
                #print(df_fold.head())
                if len(df_exmodel[fold]) == 0:
                    df_exmodel[fold] = df_fold
                    #print(df_exmodel[fold].shape)
                else:
                    df_exmodel[fold] = df_exmodel[fold].merge(df_fold, right_index = True, left_index = True)
                    #print(df_exmodel[fold].shape)
    return df_exmodel


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

def transform_categorical(df):
    """
    Convert categorical variables to Dummies using pd.get_dummies
    """
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
            print("NUMERIC: ", col)
        except Exception as e:
            df_categorical =  pd.get_dummies(df[col])
            del df[col]
            df = df.merge(df_categorical, right_index = True, left_index = True)
            print("CATEGORICAL: ", col)
    return df


def transform_KFold_random(df, y_column, K):

    df = df.reset_index()
    all_index =  df.index
    df_kfolded = dict()
    obs_per_fold = int(np.round(len(df)/K))

    for k in range(K):
        print("K: {}  all_index: {} ".format(k, len(all_index)))
        df_kfolded[k] = dict()
        df_copy = df.copy()
        if k == K-1:
            df_kfolded[k]["y"] = df_copy.loc[all_index, y_column]
            del df_copy[y_column]
            df_kfolded[k]["data"]  = df_copy.loc[all_index]
            df_kfolded[k]["index"] = all_index
            if y_column in df_kfolded[k]["data"]:
                print("\n\n\nERROR: Target variable in df_kfolded[k][data]")
        else:

            fold_index = np.random.choice(all_index, obs_per_fold, replace= False)
            fold_data  = df_copy.loc[fold_index]
            fold_y     = fold_data[y_column]
            del fold_data[y_column]

            all_index = [item for item in all_index if item not in fold_index]

            df_kfolded[k]["index"] = fold_index
            df_kfolded[k]["data"]  = fold_data
            df_kfolded[k]["y"]     = fold_y

            if y_column in df_kfolded[k]["data"]:
                print("\n\n\nERROR: Target variable in df_kfolded[k][data]")

        print( len(df_kfolded[k]["index"]))

    df_copy = df.copy()
    df_kfolded["all_data"] = dict()
    df_kfolded["all_data"]["index"] =  df.index
    df_kfolded["all_data"]["y"]     =  df_copy[y_column]
    del df_copy[y_column]
    df_kfolded["all_data"]["data"]  =  df_copy

    return df_kfolded

def transform_KFold_groups(df, y_column, K, group_id):
    """Generates a df_kfolded dictioanry. Each entry in the dictionary conatins a fold which have a 
    unique set of groups from the groups_id column that are not reapeated in any other fold.
    For example is group_id is the list od products from a store, then each fold will contain subset
    of these list and the interection of all the subsets will be null.
    Also the function have a balanced number of opservation in each fold """
    df = df.reset_index(drop = True)
    df_kfolded = dict()
    unique_vis = np.array(sorted(df[group_id].unique()))

    # Get folds
    folds = GroupKFold(n_splits=K)
    fold_ids= []
    ids = np.arange(df.shape[0])

    for trn_vis, val_vis in folds.split(X=unique_vis, y=unique_vis, groups=unique_vis):
        fold_ids.append(
            [
                ids[df[group_id].isin(unique_vis[trn_vis])],
                ids[df[group_id].isin(unique_vis[val_vis])]
            ]
        )
    for k, (trn_, val_) in enumerate(fold_ids):
        df_kfolded[k] = dict()
        df_kfolded[k]["data"]  = df.iloc[val_]
        df_kfolded[k]["index"] = val_
        df_kfolded[k]["y"]     =  df.iloc[val_][y_column]
        del df_kfolded[k]["data"][y_column]

    df_copy = df.copy()
    df_kfolded["all_data"] = dict()
    df_kfolded["all_data"]["data"]  =  df_copy
    df_kfolded["all_data"]["index"] =  df.index
    df_kfolded["all_data"]["y"]     =  df_copy[y_column]
    del df_kfolded["all_data"]["data"][y_column]

    return df_kfolded

