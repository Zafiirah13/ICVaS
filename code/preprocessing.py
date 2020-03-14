import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE,ADASYN
import numpy as np

# ----------------------------------------------------------------------------------
#               				NORMALISATION
# ----------------------------------------------------------------------------------


def normalisation(x_train,x_test,label,nFeatures, normalisation=True):
    if normalisation:
        scaler                = StandardScaler().fit(x_train.iloc[:,0:nFeatures])
        X_train_normalisation = pd.DataFrame(scaler.transform(x_train.iloc[:,0:nFeatures]))
        y_train_label         = x_train.New_label
        filename_train        = x_train.File_Name

        X_test_normalisation = pd.DataFrame(scaler.transform(x_test.iloc[:,0:nFeatures]))
        y_test_label         = x_test[label]
        filename_test        = x_test.File_Name
        
    else:
        #scaler                = StandardScaler().fit(x_train.iloc[:,0:nFeatures])
        X_train_normalisation = pd.DataFrame(x_train.iloc[:,0:nFeatures])
        y_train_label         = x_train.New_label
        filename_train        = x_train.File_Name

        X_test_normalisation = pd.DataFrame(x_test.iloc[:,0:nFeatures])
        y_test_label         = x_test[label]
        filename_test        = x_test.File_Name
        
    
    # A check to see whether the mean of x_train and X_test are ~ 0 with std 1.0
#     print(X_train_normalisation.mean(axis=0))
#     print(X_train_normalisation.std(axis=0))
#     print(X_test_normalisation.mean(axis=0))
#     print(X_test_normalisation.std(axis=0))
    
    return X_train_normalisation, y_train_label, filename_train, X_test_normalisation,\
           y_test_label, filename_test

# ----------------------------------------------------------------------------------
#                               Sigma Clipping
# ----------------------------------------------------------------------------------

def sigma_clipping(date, mag, err, threshold=3, iteration=1):
    """
    Remove any fluctuated data points by magnitudes.
    
    Parameters
    ----------
    date      : an array of dates
    mag       : an array of magnitudes
    err       : an array of magnitude errors
    threshold : float, optional (Threshold for sigma-clipping) Here we use 3 sigma
    iteration : int, optional (the number of iteration)
    
    Returns
    -------
    date : an array of Sigma-clipped dates
    mag  : an array of Sigma-clipped magnitudes.
    err  : an array of Sigma-clipped magnitude errors.
    """

    if (len(date) != len(mag)) \
        or (len(date) != len(err)) \
        or (len(mag)  != len(err)):
        raise RuntimeError('Warning message: The length of date, mag, and err must be same.')

    # By magnitudes
    for i in range(int(iteration)):
        mean  = np.median(mag)
        std   = np.std(mag)
        index = (mag >= mean - threshold*std) & (mag <= mean + threshold*std)
        date  = date[index]
        mag   = mag[index]
        err   = err[index]

    return date, mag, err


# ----------------------------------------------------------------------------------
#               				SMOTE AUGMENTATION
# ----------------------------------------------------------------------------------

def smote_augmentation(training,testing,label,nFeatures,aug_tech='ADASYN',augmentation=False):
    X_train_normalisation, y_train_np, filename_train, X_test_normalisation,\
    y_test_np, filename_test = normalisation(training,testing,label,nFeatures) 
    
    
    y_label_bf = np.unique(y_train_np)
    if augmentation:
        for i in range(len(y_label_bf)):
            print("Before OverSampling, counts of label {}: {}".format(y_label_bf[i],(y_train_np[y_train_np==y_label_bf[i]]).shape))

        if (aug_tech == 'ADASYN'):
            ada          = ADASYN(ratio ='all')#
            X_train_aug, y_train_aug = ada.fit_sample(X_train_normalisation, y_train_np.ravel())
            data_1 = pd.DataFrame(X_train_aug)
            data_1['True_class_labels'] = y_train_aug
            X_train_norm = data_1.iloc[:,0:nFeatures]
            y_train_norm = data_1.iloc[:,nFeatures]
        
        
        else:

            # sm = SMOTE(random_state=2, ratio = 1.0,kind='svm')
            sm = SMOTE(ratio = 'all')
            X_train_aug, y_train_aug = sm.fit_sample(X_train_normalisation, y_train_np.ravel())
            data_1 = pd.DataFrame(X_train_aug)
            data_1['True_class_labels'] = y_train_aug
            X_train_norm = data_1.iloc[:,0:nFeatures]
            y_train_norm = data_1.iloc[:,nFeatures]


        y_label_af = np.unique(y_train_norm)
        print('-'*70)
        for j in range(len(y_label_af)):
                print("After OverSampling, counts of label {}: {}".format(y_label_af[j],y_train_norm.loc[y_train_norm==y_label_af[j]].shape))


        X_train = X_train_norm   
        y_train = y_train_norm
        y_test  = y_test_np
        X_test  = X_test_normalisation
        
        return X_train, y_train, X_test, y_test

