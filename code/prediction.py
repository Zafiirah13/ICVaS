import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.metrics import classification_report_imbalanced
from keras.utils import np_utils
from sklearn.metrics import matthews_corrcoef, classification_report, confusion_matrix,balanced_accuracy_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc, roc_auc_score


def model_save(classifier_optimize, X_train, y_train, filename_model, save_model=False):
    fit_model      = classifier_optimize.fit(X_train, y_train)
    
    if save_model:
        pickle.dump(fit_model, open(filename_model, 'wb'))
        
    return fit_model

def model_fit(fit_model, filename_model, X_train, y_train, X_test, y_test, classifier_model='Random Forest Classifier',classes=["Type 1" , "Type 2"], filename ='./results/',load_model=False):
    if load_model:
        fit_model      = pickle.load(open(filename_model, 'rb'))
    
    else:
        fit_model = fit_model
        
    ypred          = fit_model.predict(X_test)
    probability    = fit_model.predict_proba(X_test)
    accuracy       = accuracy_score(y_test, ypred)
    MCC            = matthews_corrcoef(y_test, ypred)
    conf_mat       = confusion_matrix(y_test, ypred)
    balance_accuracy = balanced_accuracy_score(y_test, ypred)

    le             = LabelEncoder()
    labels         = le.fit_transform(y_test)
    yTest          = np_utils.to_categorical(labels,len(classes))
    auc_value      = roc_auc_score(yTest,probability)    
    misclassified  = np.where(y_test != ypred)[0]
 
    
    name_file = open(filename + ".txt", 'w')
    name_file.write('='*80+'\n')
    name_file.write('******* Testing Phase '+ str(classifier_model) +' for ' + str(classes) + ' *******\n')
    name_file.write('='*80+'\n')
    name_file.write("Accuracy: "                    + "%f" % float(accuracy) + '\n')
    name_file.write("Mathews Correlation Coef: "    + "%f" % float(MCC)      + '\n')
    name_file.write("Balanced Accuracy: "    + "%f" % float(balance_accuracy)      + '\n')
    name_file.write('='*80+'\n')
    name_file.write('='*80+'\n')
    name_file.write('Classification Report\n')
    name_file.write('='*80+'\n')
    name_file.write(classification_report(y_test, ypred, target_names = classes)+'\n')
    name_file.write('='*80+'\n')
    name_file.write('='*80+'\n')
    name_file.write('Classification Report using imabalanced metrics\n')
    name_file.write('='*80+'\n')
    name_file.write(classification_report_imbalanced(y_test, ypred, target_names = classes)+'\n')
    name_file.write('='*80+'\n')
    name_file.close()
        
    return ypred, balance_accuracy, MCC, conf_mat, misclassified


def find_misclassification(misclassified,y_test,test_data,ypred, save_dir=r'c:\data\np.txt'):
    test_data['Prediction'] = ypred
    new_DF                  = test_data.drop(test_data.index[misclassified]) # This dataset is the test set after removing the misclassification which are used in the next layer   
    misclassified_data      = test_data.iloc[misclassified] # Dataframe to store all misclassification
    misclassified_data.to_csv(save_dir, sep=' ',index=None)
    print('Test set has shape {}'.format(test_data.shape))
    print('Misclassified data has shape {}'.format(misclassified_data.shape))
    print('New test set has shape {}'.format(new_DF.shape))
    return misclassified_data, new_DF


