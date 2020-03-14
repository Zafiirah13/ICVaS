import numpy as np
import shutil
import os
import glob
import pandas as pd
import matplotlib.pylab as plt
import pickle


from sklearn.model_selection import StratifiedKFold



plt.rc('text', usetex=True)
plt.rc('font',**{'family':'serif','serif':['Palatino']})
figSize  = (12, 8)
fontSize = 20

import itertools
from scipy import interp
from itertools import cycle, islice


# Some preprocessing utilities
from sklearn.utils import shuffle
from sklearn.manifold.t_sne import TSNE
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE,ADASYN
from xgboost import XGBClassifier

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval

# The different classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import matthews_corrcoef, classification_report, confusion_matrix,balanced_accuracy_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc, roc_auc_score



def stars_label(data, label, column_name='True_class_labels'):
    '''Set variable names to specific class label'''
    stars = data[data[column_name] == label]
    return stars

# ----------------------------------------------------------------------------------
#                          First Layer Hierarchical Level
# ----------------------------------------------------------------------------------


def first_layer(contact_Bi_train, semi_det_Bi_train,rot_train,RRab_train, RRc_train, RRd_train, blazhko_train, LPV_train, delta_scuti_train, ACEP_train, cep_ii_train,\
    contact_Bi_test, semi_det_Bi_test,rot_test,RRab_test, RRc_test, RRd_test, blazhko_test, LPV_test, delta_scuti_test, ACEP_test, cep_ii_test,\
    eclipsing_label,rotational_label,pulsating_label):
    '''
    We define first layer of the hierarchical tree. The first layer consists of Eclipsing Binaries, Rotational,
    and Pulsating 
    '''
    
    # First Layer
    eclipsing_binary_train       = pd.concat([contact_Bi_train, semi_det_Bi_train], axis=0)
    eclipsing_binary_train_class = np.full(len(eclipsing_binary_train), eclipsing_label, dtype=int)

    rotational_train       = rot_train
    rotational_train_class = np.full(len(rotational_train),rotational_label, dtype=int)

    pulsating_train       = pd.concat([RRab_train, RRc_train, RRd_train, blazhko_train, LPV_train, delta_scuti_train, ACEP_train, cep_ii_train] ,axis=0)
    pulsating_train_class = np.full(len(pulsating_train), pulsating_label, dtype=int)


    print("eclipsing_binary_train has {}".format(eclipsing_binary_train.shape))
    print("pulsating_train has {}".format(pulsating_train.shape))
    print("rotational_train has {}".format(rotational_train.shape))

    eclipsing_binary_test       = pd.concat([contact_Bi_test, semi_det_Bi_test], axis=0)
    eclipsing_binary_test_class = np.full(len(eclipsing_binary_test), eclipsing_label, dtype=int)

    rotational_test       = rot_test
    rotational_test_class = np.full(len(rotational_test), rotational_label, dtype=int)

    pulsating_test       = pd.concat([RRab_test, RRc_test, RRd_test, blazhko_test, LPV_test, delta_scuti_test, ACEP_test, cep_ii_test] ,axis=0)
    pulsating_test_class = np.full(len(pulsating_test), pulsating_label, dtype=int)


    print("eclipsing_binary_test has {}".format(eclipsing_binary_test.shape))
    print("pulsating_test has {}".format(pulsating_test.shape))
    print("rotational_test has {}".format(rotational_test.shape))
    
    first_layer_train       = pd.concat([eclipsing_binary_train, rotational_train, pulsating_train], axis=0)
    first_layer_train_class = np.concatenate((eclipsing_binary_train_class, rotational_train_class, pulsating_train_class), axis=0)
    training_data_FL        = pd.DataFrame(first_layer_train)
    training_data_FL['New_label'] = first_layer_train_class
#     print(training_data_FL.shape)

    first_layer_test       = pd.concat([eclipsing_binary_test, rotational_test, pulsating_test], axis=0)
    first_layer_test_class = np.concatenate((eclipsing_binary_test_class, rotational_test_class, pulsating_test_class), axis=0)
    testing_data_FL        = pd.DataFrame(first_layer_test)
    testing_data_FL['New_label'] = first_layer_test_class
    
    y_FL_training, y_FL_training_counts = np.unique(first_layer_train_class, return_counts=True)

    
    return training_data_FL, testing_data_FL, y_FL_training_counts


# ----------------------------------------------------------------------------------
#  Second Layer Hierarchical level for first Branch: Eclipsing Binaries (Ecl & EA)
# ----------------------------------------------------------------------------------

def second_layer_EB(contact_Bi_train,semi_det_Bi_train,contact_Bi_test,semi_det_Bi_test,true_class_5,true_class_6):
    
    # Second Layer Eclipsing Binary    
    ecl_train = contact_Bi_train
    ecl_train_class = np.full(len(ecl_train), true_class_5, dtype=int)

    EA_train       = semi_det_Bi_train
    EA_train_class = np.full(len(EA_train),true_class_6, dtype=int)
 
    print("ecl train has {}".format(ecl_train.shape))
    print("EA_train has {}".format(EA_train.shape))

    ecl_test       = contact_Bi_test
    ecl_test_class = np.full(len(ecl_test), true_class_5, dtype=int)

    EA_test       = semi_det_Bi_test
    EA_test_class = np.full(len(EA_test), true_class_6, dtype=int)

    print("ecl_test has {}".format(ecl_test.shape))
    print("EA_test has {}".format(EA_test.shape))

    
    second_layer_EB_train       = pd.concat([ecl_train, EA_train], axis=0)
    second_layer_EB_train_class = np.concatenate((ecl_train_class,EA_train_class), axis=0)
    training_data_SL_EB         = pd.DataFrame(second_layer_EB_train)
    training_data_SL_EB['New_label'] = second_layer_EB_train_class
#     print(training_data_FL.shape)

    second_layer_EB_test       = pd.concat([ecl_test, EA_test], axis=0)
    second_layer_EB_test_class = np.concatenate((ecl_test_class, EA_test_class), axis=0)
    testing_data_SL_EB         = pd.DataFrame(second_layer_EB_test)
    testing_data_SL_EB['New_label'] = second_layer_EB_test_class
    
    y_SL_EB_training, y_SL_EB_training_counts = np.unique(second_layer_EB_train_class, return_counts=True)

    
    return training_data_SL_EB, testing_data_SL_EB, y_SL_EB_training_counts


# ----------------------------------------------------------------------------------
#               Second Layer Hierarchical level for 2nd Branch: RLCD
#                    RR Lyrae, LPV, Cepheid and $\delta$-Scuti
# ----------------------------------------------------------------------------------

def second_layer_RLCD(RRab_train,RRc_train,RRd_train,blazhko_train,LPV_train,ACEP_train,cep_ii_train,delta_scuti_train,\
    RRab_test,RRc_test,RRd_test,blazhko_test,LPV_test,ACEP_test, cep_ii_test,delta_scuti_test,RR_Lyrae_label,\
    LPV_label,cepheids_label,delta_scuti_label):
    
    # First Layer
    RR_Lyrae_train       = pd.concat([RRab_train,RRc_train,RRd_train,blazhko_train], axis=0)
    RR_Lyrae_train_class = np.full(len(RR_Lyrae_train), RR_Lyrae_label, dtype=int)

    LPV_train_class = np.full(len(LPV_train),LPV_label, dtype=int)

    cepheids_train       = pd.concat([ACEP_train,cep_ii_train] ,axis=0)
    cepheids_train_class = np.full(len(cepheids_train), cepheids_label, dtype=int)
    
    ds_train       = delta_scuti_train
    ds_train_class = np.full(len(ds_train), delta_scuti_label, dtype=int)


    print("RR Lyrae train has {}".format(RR_Lyrae_train.shape))
    print("LPV train has {}".format(LPV_train.shape))
    print("Cepheids train has {}".format(cepheids_train.shape))
    print("Delta Scuti train has {}".format(ds_train.shape))

    RR_Lyrae_test       = pd.concat([RRab_test,RRc_test,RRd_test,blazhko_test], axis=0)
    RR_Lyrae_test_class = np.full(len(RR_Lyrae_test), RR_Lyrae_label, dtype=int)

    LPV_test_class = np.full(len(LPV_test), LPV_label, dtype=int)

    cepheids_test       = pd.concat([ACEP_test, cep_ii_test] ,axis=0)
    cepheids_test_class = np.full(len(cepheids_test), cepheids_label, dtype=int)
    
    ds_test       = delta_scuti_test
    ds_test_class = np.full(len(ds_test), delta_scuti_label, dtype=int)


    print("RR_Lyrae_test has {}".format(RR_Lyrae_test.shape))
    print("LPV_test has {}".format(LPV_test.shape))
    print("cepheids_test has {}".format(cepheids_test.shape))
    print("Delta Scuti test has {}".format(ds_test.shape))
    
    second_layer_RLCD_train       = pd.concat([RR_Lyrae_train,LPV_train,cepheids_train,ds_train], axis=0)
    second_layer_RLCD_train_class = np.concatenate((RR_Lyrae_train_class,LPV_train_class,cepheids_train_class,ds_train_class), axis=0)
    training_data_SL_RLCD         = pd.DataFrame(second_layer_RLCD_train)
    training_data_SL_RLCD['New_label'] = second_layer_RLCD_train_class
#     print(training_data_FL.shape)

    second_layer_RLCD_test       = pd.concat([RR_Lyrae_test,LPV_test,cepheids_test,ds_test], axis=0)
    second_layer_RLCD_test_class = np.concatenate((RR_Lyrae_test_class,LPV_test_class,cepheids_test_class,ds_test_class), axis=0)
    testing_data_SL_RLCD         = pd.DataFrame(second_layer_RLCD_test)
    testing_data_SL_RLCD['New_label'] = second_layer_RLCD_test_class
    
    y_SL_RLCD_training, y_SL_RLCD_training_counts = np.unique(second_layer_RLCD_train_class, return_counts=True)

    print(y_SL_RLCD_training)
    print(y_SL_RLCD_training_counts)
    return training_data_SL_RLCD, testing_data_SL_RLCD, y_SL_RLCD_training_counts


# ----------------------------------------------------------------------------------
#            Third Layer Hierarchical level for first Branch: RRLyrae
#                          RRab, RRc, RRd, and Blazhko
# ----------------------------------------------------------------------------------

def third_layer_RRLyrae(RRab_train,RRc_train,RRd_train,blazhko_train,RRab_test,RRc_test,RRd_test,blazhko_test,\
    true_class_1,true_class_2,true_class_3,true_class_4):
    
    # Third Layer
    RRab_train_class    = np.full(len(RRab_train), true_class_1, dtype=int)
    RRc_train_class     = np.full(len(RRc_train), true_class_2, dtype=int)
    RRd_train_class     = np.full(len(RRd_train), true_class_3, dtype=int)
    blazhko_train_class = np.full(len(blazhko_train), true_class_4, dtype=int)

    print("RRab train has {}".format(RRab_train.shape))
    print("RRc train has {}".format(RRc_train.shape))
    print("RRd train has {}".format(RRd_train.shape))
    print("Blazhko train has {}".format(blazhko_train.shape))
    
    RRab_test_class    = np.full(len(RRab_test), true_class_1, dtype=int)
    RRc_test_class     = np.full(len(RRc_test), true_class_2, dtype=int)
    RRd_test_class     = np.full(len(RRd_test), true_class_3, dtype=int)
    blazhko_test_class = np.full(len(blazhko_test), true_class_4, dtype=int)

    print("RRab test has {}".format(RRab_test.shape))
    print("RRc test has {}".format(RRc_test.shape))
    print("RRd test has {}".format(RRd_test.shape))
    print("Blazhko test has {}".format(blazhko_test.shape))

    
    third_layer_RRLyrae_train       = pd.concat([RRab_train,RRc_train,RRd_train,blazhko_train], axis=0)
    third_layer_RRLyrae_train_class = np.concatenate((RRab_train_class,RRc_train_class,RRd_train_class,blazhko_train_class), axis=0)
    training_data_TL_RRLyrae        = pd.DataFrame(third_layer_RRLyrae_train)
    training_data_TL_RRLyrae['New_label'] = third_layer_RRLyrae_train_class
#     print(training_data_FL.shape)

    third_layer_RRLyrae_test       = pd.concat([RRab_test,RRc_test,RRd_test,blazhko_test], axis=0)
    third_layer_RRLyrae_test_class = np.concatenate((RRab_test_class,RRc_test_class,RRd_test_class,blazhko_test_class), axis=0)
    testing_data_TL_RRLyrae         = pd.DataFrame(third_layer_RRLyrae_test)
    testing_data_TL_RRLyrae['New_label'] = third_layer_RRLyrae_test_class
    
    y_TL_RRLyrae_training, y_TL_RRLyrae_training_counts = np.unique(third_layer_RRLyrae_train_class, return_counts=True)

    
    return training_data_TL_RRLyrae, testing_data_TL_RRLyrae, y_TL_RRLyrae_training_counts



# ----------------------------------------------------------------------------------
#            Third Layer Hierarchical level for 2nd Branch: Cepheids
#                               ACEP and Cep-II
# ----------------------------------------------------------------------------------


def third_layer_Cepheids(ACEP_train,cep_ii_train,ACEP_test,cep_ii_test,true_class_10,true_class_12):
    
    # Third Layer
    ACEP_train_class   = np.full(len(ACEP_train), true_class_10, dtype=int)
    cep_ii_train_class = np.full(len(cep_ii_train), true_class_12, dtype=int)

    print("ACEP train has {}".format(ACEP_train.shape))
    print("Cep-II train has {}".format(cep_ii_train.shape))


    ACEP_test_class   = np.full(len(ACEP_test), true_class_10, dtype=int)
    cep_ii_test_class = np.full(len(cep_ii_test), true_class_12, dtype=int)

    print("ACEP test has {}".format(ACEP_test.shape))
    print("Cep-II test has {}".format(cep_ii_test.shape))
    
    third_layer_cep_train       = pd.concat([ACEP_train,cep_ii_train], axis=0)
    third_layer_cep_train_class = np.concatenate((ACEP_train_class,cep_ii_train_class), axis=0)
    training_data_TL_cep        = pd.DataFrame(third_layer_cep_train)
    training_data_TL_cep['New_label'] = third_layer_cep_train_class
#     print(training_data_FL.shape)

    third_layer_cep_test       = pd.concat([ACEP_test,cep_ii_test], axis=0)
    third_layer_cep_test_class = np.concatenate((ACEP_test_class,cep_ii_test_class), axis=0)
    testing_data_TL_cep        = pd.DataFrame(third_layer_cep_test)
    testing_data_TL_cep['New_label'] = third_layer_cep_test_class
    
    y_TL_cep_training, y_TL_cep_training_counts = np.unique(third_layer_cep_train_class, return_counts=True)

    
    return training_data_TL_cep, testing_data_TL_cep, y_TL_cep_training_counts





