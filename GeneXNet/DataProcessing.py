"""
******************************************************************************
*** Module: GeneXNet (Gene eXpression Network) ***

*** DATA PROCESSING ***

Deep Learning for Multi-Tissue Classification of Gene Expressions (GeneXNet)

Author: Tarek_Khorshed 
Email: Tarek_Khorshed@aucegypt.edu
Version: 1.0

******************************************************************************
*** COPYRIGHT ***

All contributions by Tarek Khorshed
Copyright (c) 2020, Tarek Khorshed (Tarek_Khorshed@aucegypt.edu)
All rights reserved

*** LICENSE ***

GNU General Public License v3.0
https://github.com/Tarek-Khorshed/GeneXNet/blob/master/LICENSE
Permissions of this strong copyleft license are conditioned on making available complete source code of licensed works and modifications, 
which include larger works using a licensed work, under the same license. 
Copyright and license notices must be preserved. Contributors provide an express grant of patent rights.

******************************************************************************

"""

print ('Copyright (c) 2020, Tarek Khorshed (Tarek_Khorshed@aucegypt.edu)')   

'''
******************************************************************************
*** Module: GeneXNet (Gene eXpression Network) ***

*** DATA PROCESSING ***

******************************************************************************
'''

# *** Import statements ***
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
from time import time
import os
import math

import uuid
import datetime
import json
import copy
from collections import OrderedDict

import psutil


def Load_GE_TCGA_Data( pDataFilePath, pLabelFilePath, pNormalize=True):
    '''
    - This function reads the TCGA Data from the csv Case Data files filtered by ProjectPrimarySite.
    - The csv files have both Case Meta Data and Gene Data which is needed to extract the training data and also the labels
    - Example: TCGA_CaseData_Breast.txt
    - Data Processing is needed as follows:
        1) Drop the first column which is the index label 
        2) Copy Case Gene Data: columns 16,end  (Total 60,483 columns)
        3) Copy Case Labels: column 13
        4) Map Labels into Classes  
        5) (Zero padding) Add extra gene feature columns with zero values  to adjust the shape of the data to allow converting to a 3D shape
           Original no. of features: 60,483  -> New no. of features: 60,492 = 142 x 142 x 3 ( 9 new columns) 
    '''
    
    # pNormalize: Flag to return normalized data or raw data.    
    # Returns: DataFile, LabelFile
    
    DataPath = pDataFilePath
    LabelPath = pLabelFilePath
    
    # Load Data and Labels
    dfCancer = pd.read_csv(DataPath)
    dfCancer.drop(columns=['Unnamed: 0'],  inplace=True)   # Drop first column which has label index

     # Filter main df for DATA only with no labels
    dfCancerData = dfCancer.iloc[:,16:]  # shape ( n_samples, 60483)
    print ('Original Data Shape: ' + str(dfCancerData.shape))
    
    # Zero padding to Add extra columns with zeross (Add 9 columns)
    # New shape (:,60492)  -> reshape to (: , 142 x 142 x 3 )
    for i in range(1,10):  # 1 to 9
      dfCancerData['G'+str(i)] = pd.Series(np.zeros(dfCancerData.shape[0]), index=dfCancerData.index)    
 
    print ('New Data Shape after zero padding: ' + str(dfCancerData.shape))
     
    dfCancerData.head()    
    
    # Get labels from the Colunm SampleType 
    y_train_df = dfCancer.iloc[:,13]  # shape :,1
    # Create a dict to Map SampleType Labels into 2 classes with integer values
    ClassLabels = dict({'Solid Tissue Normal': 0, 'Primary Tumor':1, 'Recurrent Tumor':1, 'Metastatic':1, 'Additional - New Primary':1, 'Additional Metastatic':1, 'Primary Blood Derived Cancer - Peripheral Blood':1 })
    # Map SampleType into the 2 Classes to create the training labels
    y_train = np.array([ClassLabels[k] for k in y_train_df.values ])    
       

    """
    # Next prepare the data for PreProcessing    
    # Note that prepreocessing functions in sklearn assum Features are in Columns and samples in Rows
    # Transform x_train to have each feature (Gene) by Columns and Samples by rows.
    # This is the default excpected for preprocessing functions to use axis=0 i.e. by row
    # so in this case each feature (gene) will be processed accross all samples
    """   
    x_train = dfCancerData.values[:,:]
    print('Numpy Training Data Type/Shape: '+str(type(x_train)) + str(x_train.shape))
  
    """ Normalize the data after scaling """
    standard_scaler = preprocessing.StandardScaler()
    x_train_norm = standard_scaler.fit_transform(x_train)   
     
    # Check whether to return normalized data or raw data. 
    # Raw data to perform Batchnormalization in the Convnet layers
    if pNormalize:
        x = x_train_norm
    else:
        x = x_train 
  
    return x, y_train



def Load_GE_TCGA_Data_ALL_FilteredByProjectPrimarySite( pDataFilePath, pLabelFilePath, pFilterProjectPrimarySite, pNormalize=True):
    '''
    - This fn loads and prepares data for Transfer learning 
    - The fn reads the TCGA Data from the full csv Case Data file then filters by ProjectPrimarySite based on the passed parameter. .
    - The csv file has both Case Meta Data and Gene Data which is needed to extract the training data and also the labels

    - Data Processing is needed as follows:
        1) Load ALL Data for ALL ProjectPrimarySites based on the DataFilePath given as parameter
        2) Drop the first column which is the index label
        3) Filter the Data to only include the ProjectPrimarySites given as parameters (ProjectPrimarySite in filter list [] )
           Example: ['Breast', 'Colorectal', 'Head and Neck', 'Kidney', 'Liver', 'Lung', 'Prostate', 'Stomach', 'Thyroid']

        *** PROCESS DATA AS IN ORIGINAL DATA LOAD
        4) Copy Case Gene Data: columns 16,end  (Total 60,483 columns)
        5) Copy Case Labels: column 13
        6) Map Labels into classes 
        7) (Zero padding) Add extra gene feature columns with zero values  to adjust the shape of the data to allow converting to a 3D shape
           Original no. of features: 60,483  -> New no. of features: 60,492 = 142 x 142 x 3 ( 9 new columns) 

  Input parameters:
    pDataFilePath: Name of Data file for the csv Case Data files filtered by ProjectPrimarySite
    pLabelFilePath: NOT USED because labels are obtained from the Data file
    pFilterProjectPrimarySite: List of manually selected ProjectPrimarySites to filter the data based on it.
                               Example: ['Breast', 'Colorectal', 'Head and Neck', 'Kidney', 'Liver', 'Lung', 'Prostate', 'Stomach', 'Thyroid']
    pNormalize: Flag to indicate whether to normalize the data   
    
    Notes on Filtering a pandas dataframe
    - filter out rows in a dataframe with column values NA/NAN
    df_Cancer = dfCancer[(dfCancer.ProjectPrimarySite.notnull) ]
    - filter rows based on a value in a list
    df_Cancer = dfCancer[( dfCancer.ProjectPrimarySite.isin(pFilterProjectPrimarySite)  )  ]       
    - filter rows based on a value not in a list    
    df_Cancer = dfCancer[( ~dfCancer.ProjectPrimarySite.isin(pFilterProjectPrimarySite)  )  ]       
    - filter rows based on multiple conditions (We have to use bitwise operators & and | )
    df_Cancer = dfCancer[( dfCancer.ProjectPrimarySite.isin(pFilterProjectPrimarySite)  &  dfCancer.SampleType != 'Solid Tissue Normal'  )  ]       
    
    
    '''
    
    # pNormalize: Flag to return normalized data or raw data.
    
    # Returns: DataFile, LabelFile
    
    DataPath = pDataFilePath
    LabelPath = pLabelFilePath
    
   
    # Load Data and Labels
    dfCancer = pd.read_csv(DataPath)
    dfCancer.drop(columns=['Unnamed: 0'],  inplace=True)   # Drop first column which has old label index

   
    # Filter the Data to only include the ProjectPrimarySites given as parameters 
    df_Cancer = dfCancer[dfCancer.ProjectPrimarySite.isin(pFilterProjectPrimarySite)  ]    
    
    df_DataAll = df_Cancer 
    print ('ALL Data Filtered by ProjectPrimarySite - Dataframe Shape: ' + str(df_DataAll.shape))    

      # Filter main df for DATA only with no labels
    dfCancerData = df_DataAll.iloc[:,16:]  # shape 
    
    # Add extra columns with zeross (Add 9 columns)
    # New shape (:,60492)  -> reshape to (: , 142 x 142 x 3 )
    for i in range(1,10):  # 1 to 9
      dfCancerData['G'+str(i)] = pd.Series(np.zeros(dfCancerData.shape[0]), index=dfCancerData.index)    
 
    print ('New Data Shape after zero padding: ' + str(dfCancerData.shape))
     
    dfCancerData.head()    
    
   
    # Get labels from the Colunm SampleType 
    y_train_df = df_DataAll.iloc[:,13]  # shape :,1
    # Create a dict to Map SampleType Labels into classes with integer values
    ClassLabels = dict({'Solid Tissue Normal': 0, 'Primary Tumor':1, 'Recurrent Tumor':1, 'Metastatic':1, 'Additional - New Primary':1, 'Additional Metastatic':1})
    # Map SampleType into Classes to create the training labels
    y_train = np.array([ClassLabels[k] for k in y_train_df.values ])    
    
    """
    # Next prepare the data for PreProcessing    
    # Note that prepreocessing functions in sklearn assum Features are in Columns and samples in Rows
    # Transform x_train to have each feature (Gene) by Columns and Samples by rows.
    # This is the default excpected for preprocessing functions to use axis=0 i.e. by row
    # so in this case each feature (gene) will be processed accross all samples
    """
    # Convert DataFrame to a nummpy array without the first row with Gene Labels
    x_train = dfCancerData.values[:,:]
    print('Numpy Training Data Type/Shape: '+str(type(x_train)) + str(x_train.shape))

   
    """ Normalize the data after scaling """
    standard_scaler = preprocessing.StandardScaler()
    x_train_norm = standard_scaler.fit_transform(x_train)  
    
   
    # Check whether to return normalized data or raw data. 
    # Raw data for Batchnormalization in the Convnet layers
    if pNormalize:
        x = x_train_norm
    else:
        x = x_train 
  
    return x, y_train



def Load_GE_TCGA_Data_TopGenes_FilteredByProjectID( pDataFilePath, pLabelFilePath, pFilterProjectID, pGenes, pNormalize=True):
    '''
    - This fn is used to display the heatmap of samples with top mutated genes.
      The data will be read for each individual ProjectID and the gene columns will be filtered 
      based on only the top mutated Genes passed as parameter. The top mutated genes are loaded separatly before calling this fn
      
    - The csv file has both Case Meta Data and Gene Data which is needed to extract the training data and also the labels
    - Data Processing is needed as follows:
        1) Load ALL Data for a ProjectPrimarySite based on the DataFilePath given as parameter
        2) Drop the first column which is the index label    
        3) Filter the Data to only include the ProjectID given as parameters (ProjectID)
           Example: ['TCGA-BRCA']
        4) Copy Case Gene Data: columns 16,end  (Total 60,483 columns)
        5) Filter Columns based on List of Genes passed as parameter

   
  Input parameters:
    pDataFilePath: Name of Data file for the csv Case Data files filtered by ProjectPrimarySite
    pLabelFilePath: NOT USED because labels are obtained from the Data file
    pFilterProjectID: One projectID which will be used to filter the data   Example: 'TCGA-BRCA'
    pGenes: List of top mutaed genes to filter the data columns. Example: ['ENSG00000141510', 'ENSG00000155657',...]
    pNormalize: Flag to indicate whether to normalize the data   
    
    Notes on Filtering a pandas dataframe
    - filter out rows in a dataframe with column values NA/NAN
    df_Cancer = dfCancer[(dfCancer.ProjectPrimarySite.notnull) ]
    - filter rows based on a value in a list
    df_Cancer = dfCancer[( dfCancer.ProjectPrimarySite.isin(pFilterProjectPrimarySite)  )  ]       
    - filter rows based on a value not in a list    
    df_Cancer = dfCancer[( ~dfCancer.ProjectPrimarySite.isin(pFilterProjectPrimarySite)  )  ]       
    - filter rows based on multiple conditions (We have to use bitwise operators & and | )
    df_Cancer = dfCancer[( dfCancer.ProjectPrimarySite.isin(pFilterProjectPrimarySite)  &  dfCancer.SampleType != 'Solid Tissue Normal'  )  ]       
    
    
    '''
    
    # pNormalize: Flag to return normalized data or raw data. 
    
    # Returns: DataFile, LabelFile
    
    DataPath = pDataFilePath
    LabelPath = pLabelFilePath
    
   
    # Load Data and Labels
    dfCancer = pd.read_csv(DataPath)
    dfCancer.drop(columns=['Unnamed: 0'],  inplace=True)   # Drop first column which has label index

    
    # Filter the Data to only include the ProjectID given as parameters 
    df_Cancer = dfCancer[dfCancer.ProjectID.isin(pFilterProjectID)  ]    
    
    print ('Data Filtered by ProjectID - Dataframe Shape: ' + str(df_Cancer.shape))    

   
    # Get a list with the CaseIDs to return and display on the heatmap
    CaseIDList = df_Cancer['CaseID'].values.tolist()
    
  
    # Filter main df for DATA only with no labels
    dfCancerData = df_Cancer.iloc[:, 16:]  
    # Update column names with only first 15 characters without the . so that GeneIDs can be matched correctly
    # For example: ENSG00000242268.2    becomes  ENSG00000242268
    dfCancerData.rename(columns=lambda x: x[0:15], inplace=True) 
    
   
    # Filter the dataframe for only the columns in the pGenes list.
    dfCancerData = dfCancerData[pGenes] 

    
    print ('New Data Shape after column filtering: ' + str(dfCancerData.shape))
     
    """
    # Next prepare the data for PreProcessing    
    # Note that prepreocessing functions in sklearn assum Features are in Columns and samples in Rows
    # Transform x_train to have each feature (Gene) by Columns and Samples by rows.
    # This is the default excpected for preprocessing functions to use axis=0 i.e. by row
    # so in this case each feature (gene) will be processed accross all samples
    """
    # Convert DataFrame to a nummpy array without the first row with Gene Labels
    # x_train is the main array to be used for preprocessing
    x_train = dfCancerData.values[:,:]
    print('Numpy Training Data Type/Shape: '+str(type(x_train)) + str(x_train.shape))

    
    """ Normalize the data after scaling """
    standard_scaler = preprocessing.StandardScaler()
    x_train_norm = standard_scaler.fit_transform(x_train)   
    
   
    # Check whether to return normalized data or raw data. 
    # Raw data to perform Batchnormalization in the Convnet layers
    if pNormalize:
        x = x_train_norm
    else:
        x = x_train 
  
    return x, CaseIDList



def Load_GE_TCGA_Data_TopGenes_ALLData( pDataFilePath, pLabelFilePath, pGenes, pNormalize=True):
    '''
    - This fn is used to display the Cluster heatmap of ALL DATA samples with top mutated genes based on
      the Clustered Groups for each Site.
      The data will be read for all Sites and the gene columns will be filtered based on only the top mutated Genes passed
      as parameter. The top mutated genes are loaded separatly before calling this fn
      
    - The csv file has both Case Meta Data and Gene Data which is needed to extract the training data and also the labels
    - Data Processing is needed as follows:
        1) Load ALL Data based on the DataFilePath given as parameter
        2) Drop the first column which is the index label and is not needed because ID and index fields are present    
        3) Copy Case Gene Data: columns 16,end  (Total 60,483 columns)
        4) Filter Columns based on List of Genes passed as parameter

    
    
  Input parameters:
    pDataFilePath: Name of Data file for the csv Case Data files filtered by ProjectPrimarySite
    pLabelFilePath: NOT USED because labels are obtained from the Data file
    pGenes: List of top mutaed genes to filter the data columns. Example: ['ENSG00000141510', 'ENSG00000155657',...]
    pNormalize: Flag to indicate whether to normalize the data   
    
    Notes on Filtering a pandas dataframe
    - filter out rows in a dataframe with column values NA/NAN
    df_Cancer = dfCancer[(dfCancer.ProjectPrimarySite.notnull) ]
    - filter rows based on a value in a list
    df_Cancer = dfCancer[( dfCancer.ProjectPrimarySite.isin(pFilterProjectPrimarySite)  )  ]       
    - filter rows based on a value not in a list    
    df_Cancer = dfCancer[( ~dfCancer.ProjectPrimarySite.isin(pFilterProjectPrimarySite)  )  ]       
    - filter rows based on multiple conditions (We have to use bitwise operators & and | )
    df_Cancer = dfCancer[( dfCancer.ProjectPrimarySite.isin(pFilterProjectPrimarySite)  &  dfCancer.SampleType != 'Solid Tissue Normal'  )  ]       
    
    
    '''
    
    # pNormalize: Flag to return normalized data or raw data
    
    # Returns: DataFile, LabelFile
    
    DataPath = pDataFilePath
    LabelPath = pLabelFilePath
    
   
    # Load Data and Labels
    dfCancer = pd.read_csv(DataPath)
    dfCancer.drop(columns=['Unnamed: 0'],  inplace=True)   # Drop first column which has old label index

       
    print ('ALL Data - Dataframe Shape: ' + str(dfCancer.shape))    

  
    # Get a list with the CaseIDs to return and display on the heatmap
    CaseIDList = dfCancer['CaseID'].values.tolist()
    
  
    # Filter main df for DATA only with no labels
    dfCancerData = dfCancer.iloc[:, 16:]  
    # Update column names with only first 15 characters without the . so that GeneIDs can be matched correctly
    # For example: ENSG00000242268.2    becomes  ENSG00000242268
    dfCancerData.rename(columns=lambda x: x[0:15], inplace=True) 
    
   
    # Filter the dataframe for only the columns in the pGenes list.
    dfCancerData = dfCancerData[pGenes] 

    
    print ('New Data Shape after column filtering: ' + str(dfCancerData.shape))
     
    """
    # Next prepare the data for PreProcessing    
    # Note that prepreocessing functions in sklearn assum Features are in Columns and samples in Rows
    # Transform x_train to have each feature (Gene) by Columns and Samples by rows.
    # This is the default excpected for preprocessing functions to use axis=0 i.e. by row
    # so in this case each feature (gene) will be processed accross all samples
    """
    # Convert DataFrame to a nummpy array without the first row with Gene Labels
    # x_train is the main array to be used for preprocessing
    x_train = dfCancerData.values[:,:]

    print('Numpy Training Data Type/Shape: '+str(type(x_train)) + str(x_train.shape))

 
    """ Normalize the data after scaling """
    standard_scaler = preprocessing.StandardScaler()
    x_train_norm = standard_scaler.fit_transform(x_train)   
    
   
    # Check whether to return normalized data or raw data. 
    # Raw data to perform Batchnormalization in the Convnet layers
    if pNormalize:
        #x = x_train_norm
        x = x_train_robust        
    else:
        x = x_train 
  
    return x, CaseIDList


  
# *** InitData_SplitTrainTest ***  
def InitData_SplitTrainTest( px_train, py_train, pSplit=0.15 , pShuffle=True, pStratify=None ):
  '''
  - This functions splits the data into training and test based on psplit percentage. 
  - The data is shuffled using train_test_split util then finaly converted into 3D shape for convolutional networks


  Input parameters:
    px_train: All training data which has been normalized if normalize flag was used
    py_train: All training labels
    pSplit: Split for percentage for training and test data
    pShuffle: Flag to shuffle the data before splitting
    pStratify: Class label array to be used to split the data in a stratified fashion by preserving 
               the same label percentages between the train and test data as in the original data set
    
  Return paramters:
    x_train_aug, x_test_aug, y_train_aug, y_test_aug  : Training and Test data 

  ''' 

  x_train_norm, y_train =  px_train, py_train

  """ Shuffle the Training data and Labels array  """

  # Display shape information for original data
  print ('x_original:' + str(x_train_norm.shape))
  print ('y_original:' + str(y_train.shape)) 

  Train_Test_Split = pSplit
  # Put condition to remove stratify if Test size is only 1 sample
  ''' By Default shuffling is TRUE'''
  if pSplit!=1:
      # Normal split based on pSplit percentage with stratify
      x_train_norm_shuffled, x_test_norm_shuffled, y_train_shuffled, y_test_shuffled = train_test_split(
              x_train_norm, y_train, test_size = Train_Test_Split , shuffle=pShuffle , stratify=pStratify )  # By Default shuffling is TRUE
  else:
      # Split=1 (Create a test set with ony 1 sample and without stratify)
      x_train_norm_shuffled, x_test_norm_shuffled, y_train_shuffled, y_test_shuffled = train_test_split(
              x_train_norm, y_train, test_size = Train_Test_Split , shuffle=pShuffle  )  # Stratify = None by default

  ''' DATASETS '''
  ''' 
  TRAINING
  x_train_norm_shuffled
  y_train_shuffled

  TESTING
  x_test_norm_shuffled
  y_test_shuffled
  '''
  print('')
  print('Train/Test Split: '+str(pSplit))
  print ('x_train_norm_shuffled:' + str(x_train_norm_shuffled.shape))
  print ('x_test_norm_shuffled:' + str(x_test_norm_shuffled.shape))
  print ('y_train_shuffled:' + str(y_train_shuffled.shape))
  print ('y_test_shuffled:' + str(y_test_shuffled.shape))
    
   
  ''' MAIN DATASETS FOR TRAINING and TESTING '''

  ''' TRAINING '''
  x_train_aug = x_train_norm_shuffled
  y_train_aug = y_train_shuffled

  ''' TESTING '''
  x_test_aug = x_test_norm_shuffled           
  y_test_aug = y_test_shuffled


  return x_train_aug, x_test_aug, y_train_aug, y_test_aug
      
   

# *** InitData_ConvertTo3DConvShape ***  
def InitData_ConvertTo3DConvShape( px_train_aug, px_test_aug, py_train_aug, py_test_aug, pConvShape_x=100, pConvShape_y=-1, pConvShape_channels=1,  pSplit=0.15):
  '''
  This functions converts the data into a 3D Convolutional shape to prepare for training in Convolutional Neural Networks
  - The 3d shape will be (data_length, pConvShape_x, pConvShape_y , NoChannels) where pConvShape_x is fixed and pConvShape_y=-1 is calculated automatically

  Input parameters:
    px_train_aug: Training data 
    py_train_aug: Training labels
    px_test_aug: Test data 
    py_test_aug: Test Labels
    pConvShape_x: width to use when building the 3D Conv shape
    pConvShape_y: height for 3D Conv shape will be calculated automatically based on the input data shape and that width pConvShape_x
    pConvShape_channels: No of Channles for Conv Shape which will typically be 3 (large data sets) or 1 (small datasets)
    pSplit: Split for percentage for training and test data (Only needed here to avoid conversion if test data is empty)
    
  Return paramters:
    x_train_conv_aug, x_test_conv_aug, y_train_aug, y_test_aug  : Training and Test data as 3D Conv shapes

  ''' 

  ''' DATASETS '''
  #TRAINING
  x_train_aug = px_train_aug
  y_train_aug = py_train_aug
  #TESTING
  x_test_aug = px_test_aug
  y_test_aug = py_test_aug


  ''' RESHAPE Data to be 3D for Conv layers '''
  train_length =  x_train_aug.shape[0]
  test_length = x_test_aug.shape[0]

  x_train_conv = np.reshape( x_train_aug, (train_length, pConvShape_x, pConvShape_y  ,pConvShape_channels) ) 
  
  if pSplit != 0: 
    x_test_conv = np.reshape( x_test_aug, (test_length, pConvShape_x, pConvShape_y  ,pConvShape_channels) ) 
  else:
    x_test_conv =  x_test_aug  # empty array
    
  ''' MAIN DATASETS FOR TRAINING and TESTING '''

  ''' TRAINING '''
  x_train_conv_aug = x_train_conv


  ''' TESTING '''
  x_test_conv_aug = x_test_conv          

  print ('\nRESHAPE Data to be 3D for Conv layers')
  print ('x_train_conv: ' + str(x_train_conv_aug.shape))
  print ('x_test_conv: ' + str(x_test_conv_aug.shape))
  print ('y_train_aug: ' + str(y_train_aug.shape))
  print ('y_test_aug: ' + str(y_test_aug.shape))

  return x_train_conv_aug, x_test_conv_aug, y_train_aug, y_test_aug

 
# *** InitData_OverSampleMinorityClass ***  
def InitData_OverSampleMinorityClass( px_train, py_train, pMethod='SMOTE' ):
  '''
  Performs Over Sampling of the minority class to overcome any potential impact on the classification performance due to class imbalance
  Based on the method passed as parameter the data is oversampled with Class Imbalance Learning using SMOTE or ADASYN 
  
    Input parameters:
        px_train: All training data 
        py_train: All training labels
        pMethod:  Oversampling method to be used: 
                  'SMOTE': Synthetic Minority Over-sampling 
                  'ADASYN: Adaptive Synthetic Sampling   
 
    Return paramters:
       x_train_OverSampled, y_train_OverSampled
  ''' 

  from imblearn.over_sampling import SMOTE, ADASYN
    
  if pMethod=='SMOTE':
     x_train_OverSampled, y_train_OverSampled = SMOTE().fit_resample(px_train, py_train)      
  elif pMethod=='ADASYN':         
     x_train_OverSampled, y_train_OverSampled = ADASYN().fit_resample(px_train, py_train)      
  else: # Return data with NO Over Sampling
     x_train_OverSampled, y_train_OverSampled = px_train, py_train

  return x_train_OverSampled, y_train_OverSampled


  
# *** InitCrossValidationData ***  
def InitCrossValidationData( px_train, py_train, pConvShape_x=100, pConvShape_y=-1, pConvShape_channels=1,  pSplit=0.15 , pStratify=None ):
  '''
  - This functions splits the data into training and test based on psplit percentage. 
  - The data is shuffled using train_test_split util then finaly converted into 3D shape for convolutional networks
  - The 3d shape will be (data_length, pConvShape_x, pConvShape_y , 1) where pConvShape_x is fixed and pConvShape_y=-1 is calculated automatically

  Input parameters:
    px_train: All training data which has been normalized if normalize flag was used
    py_train: All training labels
    pConvShape_x: width to use when building the 3D Conv shape
    pConvShape_y: height for 3D Conv shape will be calculated automatically based on the input data shape and that width pConvShape_x
    pSplit: Split for percentage for training and test data
    pStratify: Class label array to be used to split the data in a stratified fashion by preserving 
               the same label percentages between the train and test data as in the original data set.    
    
  Return paramters:
    x_train_conv_aug, x_test_conv_aug, y_train_aug, y_test_aug  : Training and Test data as 3D Conv shapes

  ''' 

  x_train_norm, y_train =  px_train, py_train


  # Display shape information for original data
  print ('x_original:' + str(x_train_norm.shape))
  print ('y_original:' + str(y_train.shape)) 


  Train_Test_Split = pSplit

  ''' By Default shuffling is TRUE'''
  x_train_norm_shuffled, x_test_norm_shuffled, y_train_shuffled, y_test_shuffled = train_test_split(
      x_train_norm, y_train, test_size = Train_Test_Split , shuffle=True , stratify=pStratify )     # By Default shuffling is TRUE

  ''' DATASETS '''
  ''' 
  TRAINING
  x_train_norm_shuffled
  y_train_shuffled

  TESTING
  x_test_norm_shuffled
  y_test_shuffled
  '''
  print('')
  print('Train/Test Split: '+str(pSplit))
  print ('x_train_norm_shuffled:' + str(x_train_norm_shuffled.shape))
  print ('x_test_norm_shuffled:' + str(x_test_norm_shuffled.shape))
  print ('y_train_shuffled:' + str(y_train_shuffled.shape))
  print ('y_test_shuffled:' + str(y_test_shuffled.shape))
    
  ''' RESHAPE Data to be 3D for Conv layers '''
  train_length =  x_train_norm_shuffled.shape[0]
  test_length = x_test_norm_shuffled.shape[0]

  x_train_conv = np.reshape( x_train_norm_shuffled, (train_length, pConvShape_x, pConvShape_y  ,pConvShape_channels) ) 

  if Train_Test_Split != 0: 
    x_test_conv = np.reshape( x_test_norm_shuffled, (test_length, pConvShape_x, pConvShape_y  ,pConvShape_channels) ) 
  else:
    x_test_conv =  x_test_norm_shuffled  
    
  ''' MAIN DATASETS FOR TRAINING and TESTING '''

  ''' TRAINING '''
  x_train_conv_aug = x_train_conv
  y_train_aug = y_train_shuffled


  ''' TESTING '''
  x_test_conv_aug = x_test_conv           
  y_test_aug = y_test_shuffled

  print ('\nRESHAPE Data to be 3D for Conv layers')
  print ('x_train_conv: ' + str(x_train_conv.shape))
  print ('x_test_conv: ' + str(x_test_conv.shape))
  print ('y_train_aug: ' + str(y_train_aug.shape))
  print ('y_test_aug: ' + str(y_test_aug.shape))

  return x_train_conv_aug, x_test_conv_aug, y_train_aug, y_test_aug


# *** BuildCrossValidationData ***

def BuildCrossValidationData(pSessionParameters, px_train_conv_aug, px_test_conv_aug, py_train_aug, py_test_aug, pNoFolds=5):
  '''
  - Define a function to Build data for Performing K-Cross Validation Training 
  - Input is training data which has been shuffled and converted into 3D shapes
  
  Input parameters:
    px_train_conv_aug: Training data 3D conv shape
    py_train_aug: Training labels
    px_test_conv_aug: Test data 3d shape
    py_test_aug: Test Labels
    
    pNoFolds: No of training folds to use. 
  
  Return parameters:
    DataFoldsAll: Dict having all the training/val folds {all_x_train_folds, all_x_val_folds, all_y_train_folds, all_y_val_folds}
  
  '''
  print('\nBuilding Cross Validation Data... ')
  
  x_train_conv_aug = px_train_conv_aug
  
  ''' 
  If softmax activation will be used, then we need to convert the label data into categoral. This includes y_train and y_test
  '''
  NoClasses = pSessionParameters['ModelBuildParameters']['NoClasses']
  Activation = pSessionParameters['ModelBuildParameters']['Activation']  
  if Activation == 'softmax':
      print('Softmax activation - Changing label data To_Categorical... ')
      y_train_aug = to_categorical(py_train_aug, NoClasses)
      y_test_aug  = to_categorical(py_test_aug, NoClasses)
  else:
      print('Sigmoid activation - Maintaining Binary label data... ')
      y_train_aug = py_train_aug
      y_test_aug  = py_test_aug
  
  # Number of Partitions 
  k = pNoFolds

  # Number of Validation Samples
  n_val = len(x_train_conv_aug) // k

 
  x_train_folds_all=[]
  x_val_folds_all=[]
  y_train_folds_all=[]
  y_val_folds_all=[]
  
  for i in range(k):
      print('******************************************'+'\n')
      print('Creating fold # ', i+1)
      # Prepare the validation data: data from partition # k
      print ('x_val ' + str(i * n_val)+':'+ str((i + 1) * n_val) )
      print ('y_val ' + str(i * n_val)+':'+ str((i + 1) * n_val) )
      x_val_fold = x_train_conv_aug[i * n_val: (i + 1) * n_val ,:,:,:]
      y_val_fold = y_train_aug[i * n_val: (i + 1) * n_val ]

      # Prepare the training data: data from all other partitions
      print ('x_train Concat [:' + str(i * n_val)+'] + ['+ str((i + 1) * n_val)+':]' )
      print ('y_train Concat [:' + str(i * n_val)+'] + ['+ str((i + 1) * n_val)+':]' )


      x_train_fold = np.concatenate(
          [x_train_conv_aug[:i * n_val,:,:,:],
           x_train_conv_aug[(i + 1) * n_val:,:,:,:]],
          axis=0)
      y_train_fold = np.concatenate(
          [y_train_aug[:i * n_val],
           y_train_aug[(i + 1) * n_val:]],
          axis=0)


      print ('Training input shape ' + str(x_train_fold.shape))
      print ('Training target shape ' + str(y_train_fold.shape))
      
      # number formating
      #'{:{width}.{prec}f}'.format(2.71343434382, width=5, prec=2)
      
     
      print ('Val input shape ' + str(x_val_fold.shape))
      print ('Val target shape ' + str(y_val_fold.shape))
     
      x_train_folds_all.append(x_train_fold)
      x_val_folds_all.append(x_val_fold)
      y_train_folds_all.append(y_train_fold)
      y_val_folds_all.append(y_val_fold)

      print ('')
      # End for i

  # Create dict for data folds 
  DataFoldsAll = dict()  
  DataFoldsAll['x_train_folds_all'] = x_train_folds_all
  DataFoldsAll['x_val_folds_all'] = x_val_folds_all
  DataFoldsAll['y_train_folds_all'] = y_train_folds_all
  DataFoldsAll['y_val_folds_all'] = y_val_folds_all
  # Add the Test data and labels to the DataFolds dict so that they can be saved/loaded together
  DataFoldsAll['x_test'] = px_test_conv_aug
  DataFoldsAll['y_test'] = y_test_aug
  
  return DataFoldsAll
  


# *** BuildCrossValidationData ***

def BuildCrossValidationData_Fold(pSessionParameters, px_train_conv_aug, px_test_conv_aug, py_train_aug, py_test_aug, pFoldIndex=0 ):
  '''
  - Define a function to Build data for Performing K-Cross Validation Training
  - Input is training data which has been shuffled and converted into 3D shapes


  Input parameters:
    px_train_conv_aug: Training data 3D conv shape
    py_train_aug: Training labels
    px_test_conv_aug: Test data 3d shape
    py_test_aug: Test Labels
    
    pNoFolds: No of training folds to use.
    pFoldIndex: The index of the fold to be created: 0, 1, 2, 3 ,4 ( 5 Folds )
  
  Return parameters:
    DataFoldsAll: Dict having ONLY A SINGLE FOLD with training/val {all_x_train_folds, all_x_val_folds, all_y_train_folds, all_y_val_folds}
  
  '''
  print('\nBuilding Cross Validation Data... ')
  
  x_train_conv_aug = px_train_conv_aug
  
  ''' 
  If softmax activation will be used, then we need to convert the label data into categoral. This includes y_train and y_test
  '''
  NoClasses = pSessionParameters['ModelBuildParameters']['NoClasses']
  Activation = pSessionParameters['ModelBuildParameters']['Activation']  
  if Activation == 'softmax':
      print('Softmax activation - Changing label data To_Categorical... ')
      y_train_aug = to_categorical(py_train_aug, NoClasses)
      y_test_aug  = to_categorical(py_test_aug, NoClasses)
  else:
      print('Sigmoid activation - Maintaining Binary label data... ')
      y_train_aug = py_train_aug
      y_test_aug  = py_test_aug
  
  # Number of Partitions 
  k = pSessionParameters['NoFolds']

  # Number of Validation Samples
  n_val = len(x_train_conv_aug) // k

 
  x_train_folds_all=[]
  x_val_folds_all=[]
  y_train_folds_all=[]
  y_val_folds_all=[]
  
  # Set the Fold to be created based on the given index passed as parameter
  i = pFoldIndex
  
  print('******************************************'+'\n')
  print('Creating fold # ', i+1)
  # Prepare the validation data: data from partition # k
  print ('x_val ' + str(i * n_val)+':'+ str((i + 1) * n_val) )
  print ('y_val ' + str(i * n_val)+':'+ str((i + 1) * n_val) )
  x_val_fold = x_train_conv_aug[i * n_val: (i + 1) * n_val ,:,:,:]
  y_val_fold = y_train_aug[i * n_val: (i + 1) * n_val ]

  # Prepare the training data: data from all other partitions
  print ('x_train Concat [:' + str(i * n_val)+'] + ['+ str((i + 1) * n_val)+':]' )
  print ('y_train Concat [:' + str(i * n_val)+'] + ['+ str((i + 1) * n_val)+':]' )


  x_train_fold = np.concatenate(
      [x_train_conv_aug[:i * n_val,:,:,:],
       x_train_conv_aug[(i + 1) * n_val:,:,:,:]],
      axis=0)
  y_train_fold = np.concatenate(
      [y_train_aug[:i * n_val],
       y_train_aug[(i + 1) * n_val:]],
      axis=0)


  print ('Training input shape ' + str(x_train_fold.shape))
  print ('Training target shape ' + str(y_train_fold.shape))
  
  # number formating
  #'{:{width}.{prec}f}'.format(2.71343434382, width=5, prec=2)
  
 
  print ('Val input shape ' + str(x_val_fold.shape))
  print ('Val target shape ' + str(y_val_fold.shape))

  x_train_folds_all.append(x_train_fold)
  x_val_folds_all.append(x_val_fold)
  y_train_folds_all.append(y_train_fold)
  y_val_folds_all.append(y_val_fold)

  print ('')


  # Create dict for data folds 
  DataFoldsAll = dict()   
  DataFoldsAll['x_train_folds_all'] = x_train_folds_all
  DataFoldsAll['x_val_folds_all'] = x_val_folds_all
  DataFoldsAll['y_train_folds_all'] = y_train_folds_all
  DataFoldsAll['y_val_folds_all'] = y_val_folds_all
  # Add the Test data and labels to the DataFolds dict so that they can be saved/loaded together
  DataFoldsAll['x_test'] = px_test_conv_aug
  DataFoldsAll['y_test'] = y_test_aug
  
  return DataFoldsAll


def BuildCrossValidationDataStratified(px_train_conv_aug, px_test_conv_aug, py_train_aug, py_test_aug, pNoFolds=4):
  '''
  - Define a function to Build data for Performing K-Cross Validation Training 
  - Input is training data which has been shuffled and converted into 3D shapes
  
  Input parameters:
    px_train_conv_aug: Training data 3D conv shape
    py_train_aug: Training labels
    px_test_conv_aug: Test data 3d shape
    py_test_aug: Test Labels
    
    pNoFolds: No of training folds to use. 
  
  Return parameters:
    DataFoldsAll: Dict having all the training/val folds {all_x_train_folds, all_x_val_folds, all_y_train_folds, all_y_val_folds}
  
  '''
  x_train_conv_aug = px_train_conv_aug
  y_train_aug = py_train_aug
  
  # Number of Partitions 
  k = pNoFolds

  skf = StratifiedKFold(n_splits=k)

  x_train_folds_all=[]
  x_val_folds_all=[]
  y_train_folds_all=[]
  y_val_folds_all=[]
  
  i=0
  for train_index, val_index in skf.split(x_train_conv_aug, y_train_aug):
      i+=1
      print('******************************************'+'\n')
      print('Creating fold # ', i)
      # Prepare the validation data: data from partition # k
      print ('Length of Training Index ' + str(len(train_index)) )
      print ('Length of Validation Index ' + str(len(val_index)) )

     
      x_val_fold = x_train_conv_aug[val_index,:,:,:]
      y_val_fold = y_train_aug[val_index]

      x_train_fold = x_train_conv_aug[train_index,:,:,:]
      y_train_fold = y_train_aug[train_index]

      print ('Training input shape ' + str(x_train_fold.shape))
      print ('Training target shape ' + str(y_train_fold.shape))
      
      # number formating
      #'{:{width}.{prec}f}'.format(2.71343434382, width=5, prec=2)
           
      print ('Val input shape ' + str(x_val_fold.shape))
      print ('Val target shape ' + str(y_val_fold.shape))

    
      x_train_folds_all.append(x_train_fold)
      x_val_folds_all.append(x_val_fold)
      y_train_folds_all.append(y_train_fold)
      y_val_folds_all.append(y_val_fold)

      print ('')
      # End for i

  # Create dict for data folds 
  DataFoldsAll = dict()   
  DataFoldsAll['x_train_folds_all'] = x_train_folds_all
  DataFoldsAll['x_val_folds_all'] = x_val_folds_all
  DataFoldsAll['y_train_folds_all'] = y_train_folds_all
  DataFoldsAll['y_val_folds_all'] = y_val_folds_all
  # Add the Test data and labels to the DataFolds dict so that they can be saved/loaded together
  DataFoldsAll['x_test'] = px_test_conv_aug
  DataFoldsAll['y_test'] = py_test_aug
  
  return DataFoldsAll

