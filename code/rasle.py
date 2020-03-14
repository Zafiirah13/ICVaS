
import glob
import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
import FATS
import os
import sys

# Root directory of the project
ROOT_DIR = os.path.abspath("../code")
# Import code
sys.path.append(ROOT_DIR)

from hierarchicalmodel import stars_label
from preprocessing import sigma_clipping


# -------------------------------------------------------------------------------------------
#						Calculates how much each class should be augmented
# -------------------------------------------------------------------------------------------
def num_augmentation(nAugmentation, y_training_counts):    
    '''
    This section calculates the number of augmentation to be carried out for each specific class
    Parameters: nAugmentation
                Integer values. Specify the total number of samples to generate
                
                y_training_counts
                A list of total number of examples each unique class has. For instance [Type 1: 1 Type 2: 3] has a 
                list of [3905 2898]
                
    Returns: number of samples
             The number of times each class will be augmented
    '''
    number_of_samples = []
    for i in range(len(y_training_counts)):
        floatNsamples   = nAugmentation/y_training_counts[i] # Calculate the number of times the class need to be augmented - in float
        nSamples        = Decimal(str(floatNsamples)).quantize(Decimal("1"), rounding=ROUND_HALF_UP) # convert float to integer values
        total_augmented = y_training_counts[i]*nSamples
        number_of_samples.append(int(nSamples))
        print('The number of sample in Class {} is {} and is now augmented by {} times. The augmented samples are {}'.format(i,y_training_counts[i],nSamples,total_augmented))
    return number_of_samples


# -------------------------------------------------------------------------------------------
#						Sample period from true period distribution
# -------------------------------------------------------------------------------------------  

def sampling_period(true_class,file_name,num_samples,N,magnitude, magnitude_err,distribution='Normal'):
    '''
    For each augmented sample, we will assign a period that has been sampled from the true period distribution 
    of their respective class.
    
    Parameters: true_class
                The class of the variable star. Integer values varies from 0 to 13
                
                num_samples
                The number of period to sample from the distribution. In our case we use num_samples=1 as we 
                sample 1 period each time
                
                distribution
                1. Normal: The mean and the std of the true period distribution is calculated and we sample one period
                           from a normal distribution using this mean and std
                           
                2.Random: We sample randomly one period from the true period distribution
                
    Returns: new_period
             The new period for the augmented data set
    '''
    
    
    ascii_data  = pd.read_csv('../data/catalogue/Ascii_SSS_Per_Table.txt',delim_whitespace=True,names = ["SSS_ID", "File_Name", "RA", "Dec", "Period", "V_CSS", "Npts", "V_amp", "Type", "Prior_ID", "No_Name1", 'No_Name2'])
    types_stars = ascii_data[ascii_data.Type==true_class]
    
    
    crts_period = types_stars[types_stars.File_Name.values.astype(int)==int(file_name)].Period.values[0]
    crts_mean   = types_stars[types_stars.File_Name.values.astype(int)==int(file_name)].V_CSS.values[0]


    snr                = np.sqrt((1/float(N))*np.sum(((magnitude - crts_mean)/magnitude_err)**2))
    half_frequency     = 1/crts_period
    uncertainty_freq   = half_frequency*np.sqrt(2/(N*snr**2))
    uncertainty_period = 1/uncertainty_freq


    std_period = (crts_period**2)*uncertainty_freq
    
    
    mu_p  = crts_period
    std_p = std_period/3.0

    if (distribution=='Normal'):
        new_period  = np.abs(np.random.normal(mu_p, std_p, num_samples))
    
        #print('The true period is {} and sample period is {}'.format(mu,new_period))
        
    return new_period

# -------------------------------------------------------------------------------------------
#						Feature Extraction for the training set
# -------------------------------------------------------------------------------------------

def feature_extraction_training(augmented_data,file_name,data_columns,features,true_class,new_label):
    '''
    Features extraction using FATS for training set (consists both real and augmented samples)
    
    Parameters: augmented_data
                A table with the following format: Time, Std_Flux, True flux, Fake flux1, Fake flux2 ..., Fake fluxN
                
                file_name
                The name for each files - to keep track of each variable stars
                
                data_columns
                The information the data has. Either data_columns = [magnitude', 'std magnitude', 'time']
                or data_columns = ['magnitude', time']
                
                features
                The list of features to be extracted: 
                features = ['Skew', 'Mean', 'Std', 'SmallKurtosis', 'Amplitude', 'Meanvariance']
                
                true_class
                The class of the variable stars. Integer values varies from 0 to 10
                
                new_label
                The new label assigned to the aggregated classes. Integer values
                
    Returns: Feature_file
             A table that contains the six features with true period for the real examples and sampling period for 
             augmented examples. The feature file has these columns
             ['Skew','Mean','Std','SmallKurtosis','Amplitude','Meanvariance','Period','File_Name','True_class_label','New_label']
    '''
    feature_file = pd.DataFrame()
        
    #data  = pd.read_csv(augmented_data, sep=',', header=None)
    data_aug  = augmented_data
    time_     = data_aug.iloc[:,0].values
    mag_err_  = data_aug.iloc[:,1]
    mag_      = data_aug.iloc[:,2]
    N_        = data_aug.shape[0]


    
    for j in range(2, data_aug.shape[1]):
        magnitude_       = data_aug.iloc[:,j].values
        lc              = np.array([ magnitude_, time_])
        feature_extract = FATS.FeatureSpace(Data=data_columns, featureList = features)
        features_cal    = feature_extract.calculateFeature(lc)
        features_name   = features_cal.result(method='features')
        features_value  = features_cal.result(method='array')
        features_df     = pd.DataFrame(features_value.reshape(1,len(features))) 

        features_df['Period']            = sampling_period(true_class,file_name,num_samples=1,N=N_,magnitude=mag_,magnitude_err=mag_err_,distribution='Normal')
        features_df['File_Name']         = str(file_name)+'_'+str(j)
        features_df['True_class_labels'] = true_class
        features_df['New_label']         = new_label
        feature_file                     = feature_file.append(features_df)

    return feature_file


def augmentation_and_featureExtraction(data_dir, data_,data_columns,period_data,features,update_ascii_period,\
                                      number_of_samples,save_folder_training = './data/rs/training_set/'):        
    '''
    Perform data augmentation and extract features using some functionality above (for e.g number_of_samples
    and feature_extraction_training).This section finds all the filenames for each specific class, starting 
    from class 0 to class 12. For each specific class, it loads the filename, then perform data augmentation
    of this object using the number of times this object needs to be augmented. Augmentation is done by randomly
    sample from a multivariate normal distribution by considering the error bars on the flux. We take the mean 
    of gaussian as the exact magnitude and 1 sigma = the error in mag 
    
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
            file           = str(data_dir)+str(datafile)+'.dat'
            data           = pd.read_csv(file, sep=' ', header=None)
            date           = data.iloc[:,0]
            mag            = data.iloc[:,1]
            mag_err        = data.iloc[:,2]
            
            time, magnitude, std_mag = sigma_clipping(date,mag,mag_err,threshold=3,iteration=2)
            #print('length of data is {} before clipping and length of data after clipping is {}'.format(len(mag),len(magnitude)))
            N              = len(magnitude)     
            
            augmented_df = pd.DataFrame()         

            s = np.random.multivariate_normal(np.array(magnitude), np.identity(N)*np.array(std_mag)**2, number_of_samples[i])
            s = s.T
                            
            df = pd.DataFrame(s)
            df.insert(loc=0, column='0', value=time.values)
            df.insert(loc=1, column='1', value=std_mag.values)
            df.insert(loc=2, column='2', value=magnitude.values)  
            augmented_df = augmented_df.append(df)
            
            true_labelling = period_data.Type[period_data.File_Name.values == datafile].values
            features_df    = feature_extraction_training(augmented_data=augmented_df,file_name=datafile,true_class=true_labelling[0],new_label=files,data_columns=data_columns,features=features)
            feature_file   = feature_file.append(features_df)
        i += 1
    Newfeature_file     = feature_file.join(update_ascii_period.set_index('File_Name'), on='File_Name', lsuffix='_sample', rsuffix='_true')    
    newFeature_data     = Newfeature_file

    newFeature_data.Period_true.fillna(newFeature_data.Period_sample, inplace=True)
    newFeature_data_df = newFeature_data.drop(labels='Period_sample', axis=1)
    final_feature_file = newFeature_data_df[[0,1,2,3,4,5,'Period_true', 'File_Name', 'True_class_labels', 'New_label']]
    final_feature_file = final_feature_file.rename(columns={'Period_true': 'Period'})
    final_feature_file.to_csv(save_folder_training+foldername+'/Training_features.csv',index=None)#'/Type'+str(files)+'_features.csv',index=None)

    return augmented_df, final_feature_file

# -------------------------------------------------------------------------------------------
#						Feature Extraction for Test set
# -------------------------------------------------------------------------------------------

def feature_extraction_test_set(data_dir,X_testing,period_data,periods,data_columns,features,save_folder_test  = './data/rs/test_set/'):
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
        file_name_test    = X_testing.File_Name[X_testing.New_label.values == files_test]
        file_name_test = file_name_test.values
#         file_name_test = file_name_test[0:20] # Need to comment
#         print(file_name_test)
        
        for datafile_test in file_name_test:
            
            file_test      = str(data_dir)+str(datafile_test)+'.dat'
            data_test      = pd.read_csv(file_test, sep=' ', header=None)
            time_test      = data_test.iloc[:,0]
            magnitude_test = data_test.iloc[:,1]
            std_mag_test   = data_test.iloc[:,2]
            
            time_te, magnitude_te, std_mag_te = sigma_clipping(time_test,magnitude_test,std_mag_test,threshold=3,iteration=2)

 
            lc              = np.array([magnitude_te, time_te])
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


def data_sets(data_dir,ascii_files):
    files  = glob.glob(data_dir+'/*.dat')
#     files  = files[0:20] # Need to comment
#     print(len(files))

    sample_set = pd.DataFrame()

    for file in files:
        filename = os.path.basename(str(file))
        file_name = filename[0:-4]

        select_file = ascii_files[ascii_files.File_Name.values.astype(int) == int(file_name)]
        sample_set = sample_set.append(select_file)
    return sample_set




