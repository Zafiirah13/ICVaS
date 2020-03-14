

import glob
import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
import FATS
import os
import sys
from george import kernels
import george
import scipy.optimize as op

# Root directory of the project
ROOT_DIR = os.path.abspath("../code")
# Import code
sys.path.append(ROOT_DIR)



from rasle import *


def phase_data_sets(data_dir,ascii_files):
    files  = glob.glob(data_dir+'/*.txt')
#     files  = files[0:20] # Need to comment
#     print(len(files))

    sample_set = pd.DataFrame()

    for file in files:
        filename = os.path.basename(str(file))
        file_name = filename[0:-4]

        select_file = ascii_files[ascii_files.File_Name.values.astype(int) == int(file_name)]
        sample_set = sample_set.append(select_file)
    return sample_set
    

def GP_augmentation_and_featureExtraction(data_dir, data_,data_columns,period_data,features,update_ascii_period,\
                                      number_of_samples,save_folder_training = './data/GP/training_set/'):        
    '''
    Perform data augmentation using Gaussian Process and extract features using some functionality above (for e.g number_of_samples
    and feature_extraction_training).This section finds all the filenames for each specific class, starting 
    from class 0 to class 12. For each specific class, it loads the filename, then perform data augmentation
    of this object using the number of times this object needs to be augmented. Augmentation is done using
    Gaussian process and fake light curves are randomly sampled within the 3-sigma interval. 
    
    Parameters: data_dir
                The directory that contains all the raw files (.dat)
                
                data_
                The training data. Here for training data, we have given some aggregated classes a label where we
                have put them under the 'New_label' columns.
                
                data_columns
                The information the data has. Either data_columns = [magnitude', 'std magnitude', 'time']
                or data_columns = ['magnitude', time']
                
                features
                The list of features to be extracted: 
                features = ['Skew', 'Mean', 'Std', 'SmallKurtosis', 'Amplitude', 'Meanvariance']
                
                update_ascii_period
                This data file include only filename that ends with '_1' with its correcponding true period from the 
                ascii catalog. This means that this data set consists of only real examples of variable stars. We need this
                to get the true period.
                
                number_of_samples
                We use the function above 'num_augmentation()' to find the number of times each class will be augmented
                
                save_folder_training
                The directory we save the training features
                
    Returns: augmented_data
             The real and augmented data Time Real_mag Fake_mag1 Fake_mag2... Fake_magN
             
             feature_file
             The save file for features containing both real and augmented. This file will be loaded and use for 
             classification.
    '''

    foldername  = "Split_1"
    if os.path.exists(foldername):
        shutil.rmtree(foldername)
    os.makedirs(save_folder_training+foldername, 0o755)   
        
    files_trueClass = np.unique(data_.New_label.values)
#     files_trueClass = files_trueClass[0:20] # Need to comment
#     print(files_trueClass)

    i=0
    feature_file = pd.DataFrame()
    for files in files_trueClass: 
        file_name = data_.File_Name[data_.New_label.values == files]
        file_name = file_name.values
#         file_name = file_name[0:20] # Need to comment
#         print(file_name)
                
        for datafile in file_name:
            file           = str(data_dir)+str(datafile)+'.txt'
#            print(file)
            data   = np.loadtxt(file)
            time   = data[:,0]
            phase   = data[:,1]
            mag      = data[:,2]
            mag_err   = data[:,3]
            
            x, y, yerr = sigma_clipping(phase,mag,mag_err,threshold=3,iteration=2)

                        
            augmented_data = pd.DataFrame() 
                        
            kernel = np.var(y) * kernels.Matern52Kernel(metric=0.1, ndim=1)
            gp     = george.GP(kernel,fit_white_noise=True)
            comp   = gp.compute(x, yerr)
            covariance_matrix = gp.get_matrix(x)
            grad_likelihood   = gp.grad_log_likelihood(y, quiet=False)
            likelihood        = gp.log_likelihood(y, quiet=False)

            # Define the objective function (negative log-likelihood in this case).
            def nll(p):
                gp.set_parameter_vector(p)
                ll = gp.log_likelihood(y, quiet=True)
                return -ll if np.isfinite(ll) else 1e25

            # And the gradient of the objective function.
            def grad_nll(p):
                gp.set_parameter_vector(p)
                return -gp.grad_log_likelihood(y, quiet=True)
            
            # Run the optimization routine.
            p0 = gp.get_parameter_vector()
            results = op.minimize(nll, p0, jac=grad_nll, method="L-BFGS-B")

            # Update the kernel and print the final log-likelihood.
            gp.set_parameter_vector(results.x)

            #Prediction: Calculate the mean and variance
            x_pred         = x# np.linspace(min(x), max(x), 1000)#x # The coordinates where the predictive distribution should be computed.
            pred, pred_var = gp.predict(y, x_pred, return_var=True)
            samples_lc     = gp.sample_conditional(y, x_pred, number_of_samples[i])   

            df         = pd.DataFrame(np.transpose(samples_lc))
#             df.insert(loc=0, column='0', value=time)
            df.insert(loc=0, column='0', value=x_pred)  
            df.insert(loc=1, column='1', value=yerr)
            df.insert(loc=2, column='2', value=y)

            augmented_data = augmented_data.append(df)
            true_labelling = period_data.Type[period_data.File_Name.values == datafile].values
            
            features_df    = feature_extraction_training(augmented_data=augmented_data,file_name=datafile,true_class=true_labelling[0],new_label=files,data_columns=data_columns,features=features)
            feature_file   = feature_file.append(features_df)
        i += 1
    Newfeature_file     = feature_file.join(update_ascii_period.set_index('File_Name'), on='File_Name', lsuffix='_sample', rsuffix='_true')    
    newFeature_data     = Newfeature_file

    newFeature_data.Period_true.fillna(newFeature_data.Period_sample, inplace=True)
    newFeature_data_df = newFeature_data.drop(labels='Period_sample', axis=1)
    final_feature_file = newFeature_data_df[[0,1,2,3,4,5,'Period_true', 'File_Name', 'True_class_labels', 'New_label']]
    final_feature_file = final_feature_file.rename(columns={'Period_true': 'Period'})
    final_feature_file.to_csv(save_folder_training+foldername+'/Training_features.csv',index=None)#'/Type'+str(files)+'_features.csv',index=None)

    return augmented_data, final_feature_file


# -------------------------------------------------------------------------------------------
#                       Feature Extraction for Test set
# -------------------------------------------------------------------------------------------

def GP_feature_extraction_test_set(data_dir,X_testing,data_columns,period_data,periods,features,save_folder_test  = './data/GP/test_set/'):
    '''
    Perform extraction of features for test set.
    
    Parameters: data_dir
                The directory that contains all the raw files (.dat)
                
                X_testing
                The testing data. Here for testing data, we have given some aggregated classes a label where we
                have put them under the 'New_label' columns.
                
                data_columns
                The information the data has. Either data_columns = [magnitude', 'std magnitude', 'time']
                or data_columns = ['magnitude', time']
                
                features
                The list of features to be extracted: 
                features = ['Skew', 'Mean', 'Std', 'SmallKurtosis', 'Amplitude', 'Meanvariance']
                                
                number_of_samples
                We use the function above 'num_augmentation()' to find the number of times each class will be augmented
                
                save_folder_testing
                The directory we save the testing features
                
    Returns: feature_file_testSet
             The save file for features containing only real examples for test set. This file will be loaded and use for 
             classification.
    '''
#     periods           = ascii_data[['File_Name', 'Period']]
    foldername        = "Split_1"
    if os.path.exists(foldername):
        shutil.rmtree(foldername)
    os.makedirs(save_folder_test+foldername, 0o755)
    nFeatures    = len(features)
    feature_file_test = pd.DataFrame()
        
    files_trueClass_test = np.unique(X_testing.New_label.values)
#     files_trueClass_test = files_trueClass_test[0:20] # Need to comment
#     print(files_trueClass_test)
    
    
    
    feature_file_test = pd.DataFrame()
    for files_test in files_trueClass_test:
        file_name_test = X_testing.File_Name[X_testing.New_label.values == files_test]
        file_name_test = file_name_test.values
#         file_name_test = file_name_test[0:20] # Need to comment
#         print(file_name_test)
        
        for datafile_test in file_name_test:
            file_test   = str(data_dir)+str(datafile_test)+'.txt'
            data_test   = np.loadtxt(file_test)
#             time_test      = data_test[:,1]
#             magnitude_test = data_test[:,2]
#             std_mag_test   = data_test[:,3]
            
            phase_test      = data_test[:,1]
            mag_test      = data_test[:,2]
            mag_err_test   = data_test[:,3]
            
            time_test, magnitude_test, magnitude_err_test = sigma_clipping(phase_test,mag_test,mag_err_test,threshold=3,iteration=2)
            
            
            lc              = np.array([ magnitude_test, time_test])
            feature_extract = FATS.FeatureSpace(Data=data_columns, featureList = features)
            features_cal    = feature_extract.calculateFeature(lc)
            features_name   = features_cal.result(method='features')
            features_value  = features_cal.result(method='array')


            features_df_test              = pd.DataFrame(features_value.reshape(1,len(features)))            
            features_df_test['File_Name'] =  str(datafile_test)   
            true_label_test               = period_data.Type[period_data.File_Name.values == datafile_test].values
            features_df_test['True_class_labels'] = true_label_test
            features_df_test['New_label']         = files_test
            feature_file_test                     = feature_file_test.append(features_df_test)          
        
        feature_file_test['File_Name'] = feature_file_test['File_Name'].astype(int)
        feature_file_testSet = feature_file_test.join(periods.set_index('File_Name'), on='File_Name')
        feature_file_testSet = feature_file_testSet[[0,1,2,3,4,5,'Period', 'File_Name', 'True_class_labels','New_label']]
#        print(feature_file_testSet)
        feature_file_testSet.to_csv(save_folder_test+foldername+'/Test_features.csv',index=None)
    return feature_file_testSet

