"""
******************************************************************************
*** Package: GeneXNet (Gene eXpression Network) ***
*** Module: GeneXNet ***

*** FULL GeneXNet MODULE DEFINITIONS ***

Deep Learning for Multi-Tissue Classification of Gene Expressions (GeneXNet)
Author: Tarek_Khorshed 
Email: Tarek_Khorshed@aucegypt.edu
Version: 1.0

******************************************************************************
*** COPYRIGHT ***

All contributions by Tarek Khorshed
Copyright (c) 2019-2020, Tarek Khorshed (Tarek_Khorshed@aucegypt.edu)
All rights reserved

*** LICENSE ***
GNU General Public License v3.0
https://github.com/Tarek-Khorshed/GeneXNet/blob/master/LICENSE
Permissions of this strong copyleft license are conditioned on making available complete source code of licensed works and modifications, 
which include larger works using a licensed work, under the same license. 
Copyright and license notices must be preserved. Contributors provide an express grant of patent rights.

******************************************************************************

"""


# *** Import statements ***
import tensorflow as tf 
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers
from keras import backend as K
from keras.callbacks import TensorBoard
from keras import optimizers
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator    
from keras import models
from keras import layers
from keras.models import model_from_json
from keras.layers import BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.models import Model
from keras.regularizers import l2
from keras.utils import to_categorical



import numpy as np
import seaborn as sns
import pandas as pd
from pandas.io.json import json_normalize
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from scipy import interp
import matplotlib.pyplot as plt 
import matplotlib as mplib
from time import time
import os
import math

import uuid
import datetime
import json
import copy
from collections import OrderedDict

import psutil


print ('Initializing GeneXNet Module...')   
print ('Copyright (c) 2020, Tarek Khorshed (Tarek_Khorshed@aucegypt.edu)')   


# *** Define module functions ***

def Init( pPlatform = GE.cPlatform_GoogleColab , pPipelineModeFlag=GE.gPipelineModeFlag, pPipelineBatchNo= GE.gPipelineBatchNo, pLoadFromGDFlag=GE.gLoadFilesFromGoogleDriveFlag, pUploadToGDFlag=GE.gUploadFilesToGoogleDriveFlag,  pMatplotStyle=GE.gMatPlotLibStyle ):

    global gPlatform  # The keyword global is so we access the global variable
    global gLoadFilesFromGoogleDriveFlag
    global gUploadFilesToGoogleDriveFlag
    global gMatPlotLibStyle
    global gPipelineModeFlag
    global gPipelineBatchNo
    
    # Set global variables from passed parameters if they have been passed with values other then defaults
    gPlatform = pPlatform
    gLoadFilesFromGoogleDriveFlag = pLoadFromGDFlag
    gUploadFilesToGoogleDriveFlag = pUploadToGDFlag
    gPipelineModeFlag = pPipelineModeFlag
    gPipelineBatchNo = pPipelineBatchNo
    gMatPlotLibStyle = pMatplotStyle
    
    sns.set(color_codes=True)  
    mplib.style.use(pMatplotStyle)   
    
    print ('Keras version: ' + keras.__version__)
    print ('Tensorflow version: ' + tf.__version__)

    # Check for GPU
    from tensorflow.python.client import device_lib   
    def get_available_devices():  
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos]    
    print(get_available_devices()) 
    
    print ('\nPlatform: ' + gPlatform)
    print ('Pipeline Mode Flag: ' + str(gPipelineModeFlag))
    print ('Pipeline: ' + gPipelineBatchNo)
    print ('Load Files from Google Drive Flag: ' + str(gLoadFilesFromGoogleDriveFlag))
    print ('Upload Files from Google Drive Flag: ' + str(gUploadFilesToGoogleDriveFlag))
    print ('Matplot Style: ' + pMatplotStyle)

    return



'''
******************************************************************************
*** Module: GeneXNet (Gene eXpression Network) ***

*** DATA PROCESSING ***

******************************************************************************
'''


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



'''
******************************************************************************
*** Module: GeneXNet (Gene eXpression Network) ***

*** MODEL BUILDING ***

******************************************************************************
'''


def Build_Model(px_train , pSessionParameters , pTrainingParameters  ):
    # Builds a new model
    ''' 
    Input parameters:
      px_train: training data to be used to set the input shape of the model
      pSessionParameters:
      pTrainingParameters
        
    Return parameters: model

    ''' 
    ModelType = pSessionParameters['ModelType']
    print('Model Type: '+ModelType)
    
    if ModelType=='GeneXNet':
        print('GeneXNet')                
        model = Build_Model_GeneXNet(px_train , pSessionParameters , pTrainingParameters)          
    elif ModelType=='ResNet':         
        print('Building Model: ResNet')
        model = Build_Model_ResNet(px_train , pSessionParameters , pTrainingParameters)
    elif ModelType=='MobileNet':
        print('Building Model: MobileNet')                
        model = Build_Model_MobileNet(px_train , pSessionParameters , pTrainingParameters)
    elif ModelType=='DenseNet':
        print('Building Model: DenseNet')                
        model = Build_Model_DenseNet(px_train , pSessionParameters , pTrainingParameters)   
    elif ModelType=='NasNet':
        print('Building Model: NasNet')                
        model = Build_Model_NasNet(px_train , pSessionParameters , pTrainingParameters)          
    elif ModelType=='ResNeXt':
        print('Building Model: ResNeXt')                            
    else:
        print('Model Type Error')    
        print(ModelType)                    
        
    return model



# Function to Add Conv Layers to a model 
def Build_Model_AddLayers(pModel ,  pSessionParameters ):

      
    # Adds Dense Layers and Droput Layers based on pSessionParameters BuildModelParameters
    ''' 
    Input parameters:
      pModel: New model 
      pSessionParameters: ModelBuildParameters: Dict of Parameters to define how the model is built.
          DenseLayers: List of Dense Layers to be added with number of neurons for each layer. Ex: DenseLayers[128, 128]
          DropoutLayers: List of Dropout Layers to be added after each Dense Layer. Should be equal to Dense Layers. Ex: DropoutLayers[0.5, 0.5]
  
     Return parameters: model
    '''
    model = pModel
    BatchNormFlag = pSessionParameters['BatchNorm']      
    DenseLayers = pSessionParameters['ModelBuildParameters']['DenseLayers']
    DropoutLayers = pSessionParameters['ModelBuildParameters']['DropoutLayers']


    for i in range(len(DenseLayers)):
        # add dense layer
        model.add(layers.Dense(DenseLayers[i], activation='relu', name='dense_{:0>2d}'.format(i+1)))
        # Add Batchnorm if flag is set to CONV_DENSE
        if BatchNormFlag == 'CONV_DENSE':
            model.add(layers.BatchNormalization(name='batchnorm_dense_{:0>2d}'.format(i+1)))                   
        # Add dropout if not 0
        if DropoutLayers[i]!=0:
            model.add(Dropout(DropoutLayers[i], name='dropout_{:0>2d}'.format(i+1)))            
       
    return model



'''******************** GeneXNet Model Builder - Start ************************************'''

'''  *** GeneXNet Model Builder *** '''

# Function to build a GeneXNet model 
def Build_Model_GeneXNet(px_train ,  pSessionParameters , pTrainingParameters ):
    
    from keras.layers import Input
       
    # Builds a new GeneXNet model
    ''' 
    Input parameters:
      px_train: training data to be used to set the input shape of the model
      pModelBuildParameters: Dict of Parameters to define how the model is built.

    Return parameters: model

    '''    
    BatchNormFlag = pSessionParameters['BatchNorm']      
    NoClasses = pSessionParameters['ModelBuildParameters']['NoClasses']
    Activation = pSessionParameters['ModelBuildParameters']['Activation']
    IncludeTopFlag = pSessionParameters['ModelBuildParameters']['IncludeTop']
    Model = pSessionParameters['ModelBuildParameters']['Model']
    Version = pSessionParameters['ModelBuildParameters']['Version'] 

    # Flags to determine whether or not to include Dense and Residual Blocks
    DenseBlockFlags = pSessionParameters['ModelBuildParameters']['DenseBlockFlags']    
    ResBlockFlags = pSessionParameters['ModelBuildParameters']['ResBlockFlags']    
    TransitionBlockFlags = pSessionParameters['ModelBuildParameters']['TransitionBlockFlags'] 

    FirstConvFlag = pSessionParameters['ModelBuildParameters']['FirstConvFlag']     
    FirstMaxPoolFlag = pSessionParameters['ModelBuildParameters']['FirstMaxPoolFlag'] 
    FirstBNFlag = pSessionParameters['ModelBuildParameters']['FirstBNFlag'] 
   

    # For DenseBlock
    DenseBlock_GrowthRate = pSessionParameters['ModelBuildParameters']['DenseBlock_GrowthRate']    
    DenseBlock_NoBlockRepeat = pSessionParameters['ModelBuildParameters']['DenseBlock_NoBlockRepeat']    
    # For ResNet Block 
    ResBlock_NoFilters = pSessionParameters['ModelBuildParameters']['ResBlock_NoFilters']    
    ResBlock_NoBlockRepeat = pSessionParameters['ModelBuildParameters']['ResBlock_NoBlockRepeat'] 

    conv_base = GeneXNet(input_shape=(px_train.shape[1:]), 
                         DenseBlockFlags = DenseBlockFlags, ResBlockFlags = ResBlockFlags, TransitionBlockFlags=TransitionBlockFlags,
                         FirstConvFlag = FirstConvFlag, FirstBNFlag = FirstBNFlag, FirstMaxPoolFlag=FirstMaxPoolFlag,
                         DenseBlock_GrowthRate = DenseBlock_GrowthRate , DenseBlock_NoBlockRepeat= DenseBlock_NoBlockRepeat, 
                         ResBlock_NoFilters =ResBlock_NoFilters, ResBlock_NoBlockRepeat=ResBlock_NoBlockRepeat ,  
                         include_top=False , weights=None, pooling='avg', classes = NoClasses)           
            
    model = models.Sequential()
    model.add(conv_base)    


    # Check that we will not perform Finetuning to add Dense Layer otherwise it will be added later during build model Finetune
    FinetuneStartBlockPrefix = ''
    if 'Finetune_Start_BlockPrefix' in pSessionParameters.keys():
        FinetuneStartBlockPrefix = pSessionParameters['Finetune_Start_BlockPrefix']                     
    
    if len(pSessionParameters['Finetune_Trainable_Layers'])==0 and FinetuneStartBlockPrefix=='':       
        model.add(layers.Dense(NoClasses, activation=Activation, name='dense_class'))            
       
    return model


'''  *** GeneXNet Model *** '''

def GeneXNet(input_shape=None,
             DenseBlockFlags = [1,1,1,1], ResBlockFlags = [1,1,1,1], TransitionBlockFlags = [1,1,1,1], FirstConvFlag=[1], FirstBNFlag =[1], FirstMaxPoolFlag =[1],
             DenseBlock_GrowthRate = 32, DenseBlock_NoBlockRepeat=[6, 12, 24, 16] , 
             ResBlock_NoFilters=[32,64,128,256] , ResBlock_NoBlockRepeat=[3,4,6,3] ,
             preact=True, use_bias=True,  # From ResNet Code
             include_top=True,
             weights=None,
             pooling=None,
             classes=2):
    
    '''
      # - Creates the GeneXNet model architecture by alternating Dense Blocks and Residual ResNet Blocks based on parameters for ModelBuild
          The main blocks can be controlled by the passed flags as described in the parameters.
    
      Input parameters:
          input_shape: input shape 
          DenseBlockFlags: Flags to determine whether to include Dense blocks. Ex:[1,1,1,1] includes all, [0,0,0,0] includes none.
          ResBlockFlags:   Flags to determine whether to include Residual blocks. Ex:[1,1,1,1] includes all, [0,0,0,0] includes none.      
          DenseBlock_GrowthRate: Number of filters in DenseBlocks produced at each layer. Default in code = 32
                                 Note: First Dense block Conv uses 4*GrowthRate and second uses only  GrowthRate
          DenseBlock_NoBlockRepeat: No of repeatitions for each of the Dense blocks. Default = [6, 12, 24, 16]        
          ResBlock_NoFilters: No of filters in each of the Residual Blocks to be passed to the Residual stack function. Default [64,128,256,512]
          ResBlock_NoBlockRepeat: No of repeatitions for the each of the Res blocks. Default [3, 4, 6, 3]            
          preact: whether to use pre-activation or not.
          use_bias: whether to use biases for convolutional layers or not
        
          include_top: whether to include the fully-connected layer at the top of the network.
          weights: None (random initialization) or the path to the weights file to be loaded.
          pooling: optional pooling mode for feature extraction
          classes: number of classes
          
          *** Flags to configure layers before main blocks ***
          FirstConvFlag[0]: Apply the first zero pading and conv layers. 
          FirstBNFlag[0]: Apply BN and activation after conv layer. 
          FirstMaxPoolFlag[0]: : Apply Max pooling to downsample. 
                
        Returns:
            GeneXNet model 
    '''
    ge_input = layers.Input(shape=input_shape)

    bn_axis = 3 

    if FirstConvFlag[0]==1:
        x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(ge_input)  
        x = layers.Conv2D(ResBlock_NoFilters[0], 7, strides=2, use_bias=use_bias, name='conv1_conv')(x)
    else:
        x = ge_input

    if FirstBNFlag[0]==1:
        x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,name='conv1_bn')(x)
        x = layers.Activation('relu', name='conv1_relu')(x)

    if FirstMaxPoolFlag[0]==1:        
        x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x) 
        x = layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)  


    # Dense Block 1 
    if DenseBlockFlags[0]==1:
        x = GeneXNet_Dense_Block(x, DenseBlock_NoBlockRepeat[0], DenseBlock_GrowthRate, name='DENSE_conv2')

    # Residual Block 1 
    if ResBlockFlags[0]==1:
        x = GeneXNet_Residual_Block(x, ResBlock_NoFilters[0], ResBlock_NoBlockRepeat[0], name='RES_conv2')    


    if TransitionBlockFlags[0]==1:    
        x = GeneXNet_Transition_block(x, 0.5, name='pool2')     

    # Dense Block 2
    if DenseBlockFlags[1]==1:
        x = GeneXNet_Dense_Block(x, DenseBlock_NoBlockRepeat[1], DenseBlock_GrowthRate, name='DENSE_conv3')

    # Residual Block 2 
    if ResBlockFlags[1]==1:
        x = GeneXNet_Residual_Block(x, ResBlock_NoFilters[1], ResBlock_NoBlockRepeat[1], name='RES_conv3')    


    if TransitionBlockFlags[1]==1:       
        x = GeneXNet_Transition_block(x, 0.5, name='pool3')

    # Dense Block 3
    if DenseBlockFlags[2]==1:
        x = GeneXNet_Dense_Block(x, DenseBlock_NoBlockRepeat[2], DenseBlock_GrowthRate, name='DENSE_conv4')

    
    # Residual Block 3 
    if ResBlockFlags[2]==1:
        x = GeneXNet_Residual_Block(x, ResBlock_NoFilters[2], ResBlock_NoBlockRepeat[2], name='RES_conv4')        
  
    

    if TransitionBlockFlags[2]==1:       
        x = GeneXNet_Transition_block(x, 0.5, name='pool4')

    # Dense Block 4 
    if DenseBlockFlags[3]==1:
        x = GeneXNet_Dense_Block(x, DenseBlock_NoBlockRepeat[3], DenseBlock_GrowthRate, name='DENSE_conv5')

    
    # Residual Block 4 
    if ResBlockFlags[3]==1:
        x = GeneXNet_Residual_Block(x, ResBlock_NoFilters[3], ResBlock_NoBlockRepeat[3], name='RES_conv5')    

        
    if TransitionBlockFlags[3]==1:       
        x = GeneXNet_Transition_block(x, 0.5, name='pool5')        
              
    if preact is True:
        x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,name='post_bn')(x)
        x = layers.Activation('relu', name='post_relu')(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(classes, activation='softmax', name='fc')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D(name='max_pool')(x)

    # Create GeneXNet model
    model = models.Model(ge_input, x, name='GeneXNet')

    return model


''' ******************* GeneXNet Residual Blocks - Start ************************************ '''

def GeneXNet_Residual_Block_Layers(x, filters, kernel_size=3, stride=1, conv_shortcut=False, name=None):
    ''' GeneXNet Residual Block Layers
    # Parameters
        x: input Gene Expressions
        filters: no of filters of the bottleneck layer
        kernel_size: kernel size of the bottleneck layer.
        stride: Stride of the first layer.
        conv_shortcut: Flag to use convolution shortcut if True, otherwise identity shortcut.
        name: Block label.
    # Returns
        Output Gene Expressions
    '''
    bn_axis = 3  # channels last 

    preact = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                       name=name + '_preact_bn')(x)
    preact = layers.Activation('relu', name=name + '_preact_relu')(preact)

    if conv_shortcut is True:
        shortcut = layers.Conv2D(4 * filters, 1, strides=stride,name=name + '_0_conv')(preact)
    else:
        shortcut = layers.MaxPooling2D(1, strides=stride)(x) if stride > 1 else x

    x = layers.Conv2D(filters, 1, strides=1, use_bias=False, name=name + '_1_conv')(preact)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
    x = layers.Conv2D(filters, kernel_size, strides=stride, use_bias=False, name=name + '_2_conv')(x)
    
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    x = layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)

    x = layers.Add(name=name + '_out')([shortcut, x])
    return x


def GeneXNet_Residual_Block(x, filters, blocks, stride1=2, name=None):
    """ GeneXNet Residual Block
    # Parameters
        x: input Gene Expressions
        filters: No of filters of the bottleneck layer in a block
        blocks: No of blocks in the stacked blocks
        stride1: stride of the first layer in the first block
        name: Block label
    # Returns
        Output Gene Expressions
    """
    x = GeneXNet_Residual_Block_Layers(x, filters, conv_shortcut=True, name=name + '_block1')        
    for i in range(2, blocks):
        x = GeneXNet_Residual_Block_Layers(x, filters, name=name + '_block' + str(i))  
    if blocks!=1:  
        x = GeneXNet_Residual_Block_Layers(x, filters, stride=stride1, name=name + '_block' + str(blocks))
    
    return x




''' ******************* GeneXNet Residual Blocks - End ************************************'''

''' ******************* GeneXNet Dense Blocks - Start ************************************ '''

def GeneXNet_Dense_Block(x, blocks, growth_rate, name):
    """GeneXNet dense block.
    # Parameters
        x: Input Gene Expressions
        blocks: Number of building blocks.
        growth_rate: growth rate
        name: Block label.
    # Returns
        output Gene Expressions
    """
    for i in range(blocks):
        x = GeneXNet_Dense_conv_block(x, growth_rate, name=name + '_block' + str(i + 1))
    return x

def GeneXNet_Dense_conv_block(x, growth_rate, name):
    """ Building block for GeneXNet Dense block.
    # Parameters
        x: Input Gene Expressions
        growth_rate: growth rate at dense layers.
        name: Block label.
    # Returns
        Output Gene Expressions
    """
    bn_axis = 3 
    x1 = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(x)
    x1 = layers.Activation('relu', name=name + '_0_relu')(x1)
    x1 = layers.Conv2D(4 * growth_rate, 1,use_bias=False, name=name + '_1_conv')(x1)
    x1 = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x1)
    x1 = layers.Activation('relu', name=name + '_1_relu')(x1)
    x1 = layers.Conv2D(growth_rate, 3, padding='same', use_bias=False, name=name + '_2_conv')(x1)
    x = layers.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x


'''******************** GeneXNet Dense Blocks - END ************************************'''

'''******************** GeneXNet Transition Blocks - Start ************************************'''

def GeneXNet_Transition_block(x, reduction, name):
    
    
    from keras import backend as K    
    
    """ GeneXNet transition block.
    # Parameters
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    bn_axis = 3 
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_bn')(x)
    x = layers.Activation('relu', name=name + '_relu')(x)
    x = layers.Conv2D(int(K.int_shape(x)[bn_axis] * reduction), 1, use_bias=False, name=name + '_conv')(x)
    x = layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x

'''******************** GeneXNet Transition Blocks - End ************************************'''


def Build_Model_LRSchedule(pEpoch, plr):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 50 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Parameters
        epoch: number of epochs
    # Returns
        lr: learning rate
    """
    
    lr = 1e-4

    if pEpoch > 350-1:
        lr *=  math.pow(0.5, 6)   # 2e-6    
    elif pEpoch > 300-1:
        lr *= math.pow(0.5, 5)   # 3e-6    
    elif pEpoch > 250-1:
        lr *= math.pow(0.5, 4)   # 6e-6
    elif pEpoch > 200-1:
        lr *= math.pow(0.5, 3)   # 1e-5
    elif pEpoch > 150-1:
        lr *= math.pow(0.5, 2)   # 3e-5 
    elif pEpoch > 100-1:     
        lr *= math.pow(0.5, 1)   # 5e-5
    

    return lr




# Function to adjust the model for Finetuning by freezing the conv layers 
def Build_Model_Finetune_GeneXNet(pmodel , pSessionParameters ):
    # Builds a new model for Finetuning
    ''' 
    Input parameters:
      pmodel: an existing model which has already been trained
      pTrainable_layers: list of layer names to be trainable which would typically be the classifier top

    Return parameters: new updated model

    ''' 
    print ("\nUpdating model for GeneXNet Finetuning...")
    print('Original Model:')
    pmodel.summary()
    
    # Add Additional Layers for Finetuning
    pmodel = Build_Model_AddLayers(pmodel , pSessionParameters )
    
    # Add a new Dense Layer
    NoClasses = pSessionParameters['ModelBuildParameters']['NoClasses']
    Activation = pSessionParameters['ModelBuildParameters']['Activation']    
    pmodel.add(layers.Dense(NoClasses, activation=Activation, name='dense_class')) 
  
    # Freeze ALl Model layers
    
    print ("\nFrozen Layers: ")   
    submodel = pmodel.layers[0]

    for layer in submodel.layers:
        print (layer.name)
        layer.trainable = False
        
           
    # unfreeze Trainable Layers starting at a certin Block 
    print ("\nTrainable Layers: ")        
    FinetuneStartBlockPrefix = pSessionParameters['Finetune_Start_BlockPrefix'] 
   
    CountLayers = 0
    StartFinetuneLayer =0
    for layer in submodel.layers:
        CountLayers+=1
        if FinetuneStartBlockPrefix in layer.name:
          print (layer.name)
          if StartFinetuneLayer==0:
            StartFinetuneLayer =CountLayers
    
    
    print()
    print('Starting Finetuning at Model Layer No: '+str(StartFinetuneLayer))
    for layer in submodel.layers[StartFinetuneLayer:]:
      layer.trainable =  True
      print (layer.name)    
    
    print('Model After Setting Finetune Layers:')
    
    for layer in pmodel.layers:
        print (layer.name)
        print('Layer status: '+str(layer.trainable))     
    print()
    
    return pmodel   # return updated model


'''******************** GeneXNet Model Builder - END ************************************'''




'''******************** Other CNN Models - Start ************************************'''

# Function to build a ResNet model 
def Build_Model_ResNet(px_train ,  pSessionParameters , pTrainingParameters ):
    
    from keras.applications.resnet import ResNet50
    from keras.applications.resnet import ResNet101
    from keras.applications.resnet import ResNet152
    
    from keras.applications.resnet_v2 import ResNet50V2
    from keras.applications.resnet_v2 import ResNet101V2
    from keras.applications.resnet_v2 import ResNet152V2
    

    
    from keras.layers import Input
       
    # Builds a new ResNet model
    ''' 
    Input parameters:
      px_train: training data to be used to set the input shape of the model
      pModelBuildParameters: Dict of Parameters to define how the model is built.

    Return parameters: model

    '''    
    BatchNormFlag = pSessionParameters['BatchNorm']     
 
    NoClasses = pSessionParameters['ModelBuildParameters']['NoClasses']
    Activation = pSessionParameters['ModelBuildParameters']['Activation']
    IncludeTopFlag = pSessionParameters['ModelBuildParameters']['IncludeTop']
    Model = pSessionParameters['ModelBuildParameters']['Model']
    Version = pSessionParameters['ModelBuildParameters']['Version'] 
    
    if IncludeTopFlag:
        if Version==1:
            if 'ResNet50' in Model:                
                conv_base = ResNet50(input_shape=(px_train.shape[1:]), weights=None, include_top=True , classes = NoClasses, pooling='avg')           
            elif 'ResNet101' in Model:
                conv_base = ResNet101(input_shape=(px_train.shape[1:]), weights=None, include_top=True , classes = NoClasses, pooling='avg')           
            elif 'ResNet152' in Model:
                conv_base = ResNet152(input_shape=(px_train.shape[1:]), weights=None, include_top=True , classes = NoClasses, pooling='avg')           
        else: # Version 2
            if 'ResNet50' in Model:                
                conv_base = ResNet50V2(input_shape=(px_train.shape[1:]), weights=None, include_top=True , classes = NoClasses, pooling='avg')           
            elif 'ResNet101' in Model:
                conv_base = ResNet101V2(input_shape=(px_train.shape[1:]), weights=None, include_top=True , classes = NoClasses, pooling='avg')           
            elif 'ResNet152' in Model:
                conv_base = ResNet152V2(input_shape=(px_train.shape[1:]), weights=None, include_top=True , classes = NoClasses, pooling='avg')           
                        
        model = models.Sequential()
        model.add(conv_base)    
    else:
        if Version==1:
            if 'ResNet50' in Model:                
                conv_base = ResNet50(input_shape=(px_train.shape[1:]), weights=None, include_top=False , classes = NoClasses, pooling='avg')           
            elif 'ResNet101' in Model:
                conv_base = ResNet101(input_shape=(px_train.shape[1:]), weights=None, include_top=False , classes = NoClasses, pooling='avg')           
            elif 'ResNet152' in Model:
                conv_base = ResNet152(input_shape=(px_train.shape[1:]), weights=None, include_top=False , classes = NoClasses, pooling='avg')           
        else: # Version 2
            if 'ResNet50' in Model:                
                conv_base = ResNet50V2(input_shape=(px_train.shape[1:]), weights=None, include_top=False , classes = NoClasses, pooling='avg')           
            elif 'ResNet101' in Model:
                conv_base = ResNet101V2(input_shape=(px_train.shape[1:]), weights=None, include_top=False , classes = NoClasses, pooling='avg')           
            elif 'ResNet152' in Model:
                conv_base = ResNet152V2(input_shape=(px_train.shape[1:]), weights=None, include_top=False , classes = NoClasses, pooling='avg')           
                
        model = models.Sequential()
        model.add(conv_base)    

        # Add Dense Layers and Dropout and BatchNorm layers based on ModelBuildParameters
        model = Build_Model_AddLayers(model , pSessionParameters )
        model.add(layers.Dense(NoClasses, activation=Activation, name='dense_class'))        
    
    
    return model


# Function to build a ResNet model 
def Build_Model_ResNeXt(px_train ,  pSessionParameters , pTrainingParameters ):

    from keras.applications.resnext import ResNeXt50
    from keras.applications.resnext import ResNeXt101    
    
    from keras.layers import Input
       
    # Builds a new ResNet model
    ''' 
    Input parameters:
      px_train: training data to be used to set the input shape of the model
      pModelBuildParameters: Dict of Parameters to define how the model is built.

    Return parameters: model

    '''    
    BatchNormFlag = pSessionParameters['BatchNorm']     

    NoClasses = pSessionParameters['ModelBuildParameters']['NoClasses']
    Activation = pSessionParameters['ModelBuildParameters']['Activation']
    IncludeTopFlag = pSessionParameters['ModelBuildParameters']['IncludeTop']
    Model = pSessionParameters['ModelBuildParameters']['Model']
    Version = pSessionParameters['ModelBuildParameters']['Version'] 
    
     
    
    if IncludeTopFlag: 
        if 'ResNeXt50' in Model:                
            conv_base = ResNeXt50(input_shape=(px_train.shape[1:]), weights=None, include_top=True , classes = NoClasses, pooling='avg')           
        elif 'ResNeXt101' in Model:
            conv_base = ResNeXt101(input_shape=(px_train.shape[1:]), weights=None, include_top=True , classes = NoClasses, pooling='avg')           
                        
        model = models.Sequential()
        model.add(conv_base)    
 
    else:
        if 'ResNeXt50' in Model:                
            conv_base = ResNeXt50(input_shape=(px_train.shape[1:]), weights=None, include_top=False , classes = NoClasses, pooling='avg')           
        elif 'ResNeXt101' in Model:
            conv_base = ResNeXt101(input_shape=(px_train.shape[1:]), weights=None, include_top=False , classes = NoClasses, pooling='avg')           
                
        model = models.Sequential()
        model.add(conv_base)    

        # Add Dense Layers and Dropout and BatchNorm layers based on ModelBuildParameters
        model = Build_Model_AddLayers(model , pSessionParameters )
        model.add(layers.Dense(NoClasses, activation=Activation, name='dense_class'))            
    
   
    return model


# Function to build a MobileNet model 
def Build_Model_MobileNet(px_train ,  pSessionParameters , pTrainingParameters ):
    
    from keras.applications.mobilenet import MobileNet
    from keras.applications.mobilenet_v2 import MobileNetV2
    from keras.layers import Input
       
    # Builds a new model
    ''' 
    Input parameters:
      px_train: training data to be used to set the input shape of the model
      pModelBuildParameters: Dict of Parameters to define how the model is built.

    Return parameters: model

    '''    
    BatchNormFlag = pSessionParameters['BatchNorm']     
 
    NoClasses = pSessionParameters['ModelBuildParameters']['NoClasses']
    Activation = pSessionParameters['ModelBuildParameters']['Activation']
    IncludeTopFlag = pSessionParameters['ModelBuildParameters']['IncludeTop']
    Model = pSessionParameters['ModelBuildParameters']['Model']
    Version = pSessionParameters['ModelBuildParameters']['Version']    
    
   
    if IncludeTopFlag:
        if Version==1:
            conv_base = MobileNet(input_shape=(px_train.shape[1:]), weights=None, include_top=True , classes = NoClasses, pooling='avg')           
        else:
            conv_base = MobileNetV2(input_shape=(px_train.shape[1:]), weights=None, include_top=True , classes = NoClasses, pooling='avg')           
        model = models.Sequential()
        model.add(conv_base)    
    else:
        if Version==1:
            conv_base = MobileNet(input_shape=(px_train.shape[1:]), weights=None, include_top=False , classes = NoClasses, pooling='avg')           
        else:
            conv_base = MobileNetV2(input_shape=(px_train.shape[1:]), weights=None, include_top=False , classes = NoClasses, pooling='avg')           
        model = models.Sequential()
        model.add(conv_base)    
 

        # Add Dense Layers and Dropout and BatchNorm layers based on ModelBuildParameters
        model = Build_Model_AddLayers(model , pSessionParameters )
        model.add(layers.Dense(NoClasses, activation=Activation, name='dense_class'))        
        
    return model



# Function to build a DenseNet model 
def Build_Model_DenseNet(px_train ,  pSessionParameters , pTrainingParameters ):
    
    from keras.layers import Input
    from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201    
       
    # Builds a new model
    ''' 
    Input parameters:
      px_train: training data to be used to set the input shape of the model
      pModelBuildParameters: Dict of Parameters to define how the model is built.

    Return parameters: model

    '''    
    BatchNormFlag = pSessionParameters['BatchNorm']     
 
    NoClasses = pSessionParameters['ModelBuildParameters']['NoClasses']
    Activation = pSessionParameters['ModelBuildParameters']['Activation']
    IncludeTopFlag = pSessionParameters['ModelBuildParameters']['IncludeTop']
    Model = pSessionParameters['ModelBuildParameters']['Model']
    
    if IncludeTopFlag:
        if 'DenseNet121' in Model:
            conv_base = DenseNet121(input_shape=(px_train.shape[1:]), weights=None, include_top=True , classes = NoClasses, pooling='avg')           
        elif 'DenseNet169' in Model:
            conv_base = DenseNet169(input_shape=(px_train.shape[1:]), weights=None, include_top=True , classes = NoClasses, pooling='avg')                       
        elif 'DenseNet201' in Model:
            conv_base = DenseNet201(input_shape=(px_train.shape[1:]), weights=None, include_top=True , classes = NoClasses, pooling='avg')                       

        model = models.Sequential()
        model.add(conv_base)    
    else:
        if 'DenseNet121' in Model:
            conv_base = DenseNet121(input_shape=(px_train.shape[1:]), weights=None, include_top=False , classes = NoClasses, pooling='avg')           
        elif 'DenseNet169' in Model:
            conv_base = DenseNet169(input_shape=(px_train.shape[1:]), weights=None, include_top=False , classes = NoClasses, pooling='avg')                       
        elif 'DenseNet201' in Model:
            conv_base = DenseNet201(input_shape=(px_train.shape[1:]), weights=None, include_top=False , classes = NoClasses, pooling='avg')                       

        model = models.Sequential()
        model.add(conv_base)    
        # Add Dense Layers and Dropout and BatchNorm layers based on ModelBuildParameters
        model = Build_Model_AddLayers(model , pSessionParameters )
        model.add(layers.Dense(NoClasses, activation=Activation, name='dense_class'))        
   
  
    return model


# Function to build a NasNet model 
def Build_Model_NasNet(px_train ,  pSessionParameters , pTrainingParameters ):
    
    from keras.layers import Input   
    from keras.applications.nasnet import NASNetLarge, NASNetMobile
       
    # Builds a new model
    ''' 
    Input parameters:
      px_train: training data to be used to set the input shape of the model
      pModelBuildParameters: Dict of Parameters to define how the model is built.

    Return parameters: model

    '''    
    BatchNormFlag = pSessionParameters['BatchNorm']     

    NoClasses = pSessionParameters['ModelBuildParameters']['NoClasses']
    Activation = pSessionParameters['ModelBuildParameters']['Activation']
    IncludeTopFlag = pSessionParameters['ModelBuildParameters']['IncludeTop']
    Model = pSessionParameters['ModelBuildParameters']['Model']

    
    if IncludeTopFlag:
        if Model == 'NasNetLarge':
            print('Building Model: NASNetLarge')                        
            conv_base = NASNetLarge(input_shape=(px_train.shape[1:]), weights=None, include_top=True , classes = NoClasses, pooling='avg')           
        elif Model == 'NasNetMobile':
            print('Building Model: NASNetMobile')                        
            conv_base = NASNetMobile(input_shape=(px_train.shape[1:]), weights=None, include_top=True , classes = NoClasses, pooling='avg')                       

        model = models.Sequential()
        model.add(conv_base)    
    else:
        if Model == 'NasNetLarge':
            print('Building Model: NASNetLarge')                        
            conv_base = NASNetLarge(input_shape=(px_train.shape[1:]), weights=None, include_top=True , classes = NoClasses, pooling='avg')           
        elif Model == 'NasNetMobile':
            print('Building Model: NASNetMobile')                        
            conv_base = NASNetMobile(input_shape=(px_train.shape[1:]), weights=None, include_top=True , classes = NoClasses, pooling='avg')                       

        model = models.Sequential()
        model.add(conv_base)    
        model = Build_Model_AddLayers(model , pSessionParameters )
     
        model.add(layers.Dense(NoClasses, activation=Activation, name='dense_class'))        
      
    return model


'''******************** Other CNN Models - End ************************************'''


# ****************************************************************************************************************************




'''
******************************************************************************
*** Module: GeneXNet (Gene eXpression Network) ***

*** MODEL TRAINING ***

******************************************************************************
'''
  
# *** TrainCrossValidation ***

def TrainCrossValidation( pDataFoldsAll, pSessionParameters , pTrainingParameters, pAugParameters , pTrainingSessionResults=None):  
  ''' 
  # Performs training using cross validation on all the given folds
  
  Input parameters:
    pDataFoldsAll: dict with all Train, Val datafolds as lists and Test data as numpy arrays
                   {x_train_folds_all, x_val_folds_all, y_train_folds_all, y_val_folds_all, x_test, y_test}
    px_test_conv_aug: Test data  (included in the DataFolds dict)
    py_test_aug: Test labels (included in the DataFolds dict)
    pNoFolds: number of training folds
    pSessionParameters: dict with session parameters. 
              SessionParameters = dict(TrainingMode='NEW',       # 1) NEW 2) RESUME 3) FINETUNE  4) RESUME_FINETUNE
                                       Finetune_Trainable_Layers=[]    # Layers to be set as trainable during finetuning
                                       ModelBuildParameters =''   # params to define how the model layers will be built 
                                       BatchNorm = 'NONE'   # Flag to indocate whether or not to add batch norm in the model
                                                              1) NONE 2) CONV_ONLY 3) CONV_DENSE (Both Conv and Dense layers)
    ***                               )     
    TrainingMode:  Text indicating training mode:  1) NEW 2) RESUME 3) FINETUNE  4) RESUME_FINETUNE
                   Text Flag to indicate whether to create new models or use previous models which were trained already and available in the pTrainingSessionResults 
                   This is needed to continue more training epochs from previous training sessions.
                   1) New: A new training model will be created and a new trainig session is executed (Similar to previous pInitModelsFlag=True)
                   2) Resume: This will resume the previous trainings using the models and dataaug data passed on from previous sessions in the pTrainingSessionResults
                   3) Finetune: This will finetune the existing models and fine tune the existing training sessions. The previous models are needed which will be obtained 
                                also from the parameter pTrainingSessionResults.But the previous dataaug structures are not needed because finetune does not use data augmentation.                                
                   4) Resume_Finetune: This will resume the finetunine of a previous finetune sessions. 
    ***
    
    pTrainingParameters: dict with training parameters: Example
          TrainingParameters = dict(
                                NoEpochs=100,  # Number of training epochs
                                BatchSize= 10  # training batch size
                                Loss='binary_crossentropy'
                                Optimizers='Adam'
                                LR =1e-4
                                Metrics='acc',
                                Verbose=1)
    pAugFlag: To indicate if data augmentation will be used
    pAugParameters: dict with augmentation parameters: Example
          AugParameters = dict(
                                featurewise_center=False,  # set input mean to 0 over the dataset
                                samplewise_center=False,  # set each sample mean to 0
                                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                                samplewise_std_normalization=False,  # divide each input by its std
                                zca_whitening=False,  # apply ZCA whitening
                                rotation_range=5,  # randomly rotate in the range (degrees, 0 to 180)
                                width_shift_range=0.05,  # randomly shift horizontally (fraction of total width)
                                height_shift_range=0.05,  # randomly shift vertically (fraction of total height)
                                shear_range=0.05,
                                zoom_range=0,
                                horizontal_flip=True,  # randomly flip 
                                vertical_flip=False, # randomly flip 
                                fill_mode='nearest')   
    pTrainingSessionResults: All training results and structures from previous training to be used in the mode of resuming a previous training.
    
    

  Return parameters: 
       TrainingSessionResults: dict with the following lists:
                               all_models: all trained models for all data folds
                               all_history: training history of all data folds
                               all_scores_training: evaluation of training scores for all folds 
                               all_scores_validation: evaluation of validation scores for all folds          
                               all_models: all_models (to be used to continue training session if needed)
                               all_datagen: all data generators (to be used to continue training session if needed)
       
  '''
  all_scores = []
  all_history = []
  all_models = []
  all_datagen = []
  all_predictions = [] 
  all_resultsinfo = []

  # Get Training and Validation data folds from the input parameter dict
  all_x_train_folds = pDataFoldsAll['x_train_folds_all']   # List with all training data folds 
  all_x_val_folds = pDataFoldsAll['x_val_folds_all']       # List with all validation data folds 
  all_y_train_folds = pDataFoldsAll['y_train_folds_all']   # List with all training label folds 
  all_y_val_folds = pDataFoldsAll['y_val_folds_all']       # List with all validation label folds 
   
  # Get lists which has Weights FileNames and GoogleDriveIDs if the Weights will be loaded
  # Check if LoadWeights is part of the ModelBuildParameters 
  if 'LoadWeights' in pSessionParameters['ModelBuildParameters'].keys():
      if pSessionParameters['ModelBuildParameters']['LoadWeights']:
          # Get the list which has the Weight file information to be loaded
          WeightsFileName = pSessionParameters['ModelBuildParameters']['WeightsFileName']   # List of FileNames for each fold
          WeightsFileGoogleDriveID = pSessionParameters['ModelBuildParameters']['WeightsFileGoogleDriveID'] # List of GoogleDriveIDs for each fold          
  
  
  # Get number of training folds (Number of Partitions )  
  k = len(all_x_train_folds) 
  
  # Loop for all training folds
  for i in range(k):
      print('\n'+'******************************************'+'\n')
      print(psutil.virtual_memory())
      print('Training fold # ', i+1)    
    
      # Get the training and validation data for fold k from the fold lists
      x_train_fold = all_x_train_folds[i]
      x_val_fold = all_x_val_folds[i]
      y_train_fold = all_y_train_folds[i]
      y_val_fold = all_y_val_folds[i]
    
    
    
      print ('Training input shape ' + str(x_train_fold.shape))
      print ('Training target shape ' + str(y_train_fold.shape))
      print ('Val input shape ' + str(x_val_fold.shape))
      print ('Val target shape ' + str(y_val_fold.shape))

      print ('')

      ' Get session parameters for training mode and data augmentation'
      TrainingMode = pSessionParameters['TrainingMode']
      AugFlag = pSessionParameters['DataAug']
      

      # Check if we will build a new model or continue training of a previous model
      if TrainingMode == "NEW":     
        # Create a new model
        print ('New training model created...')
        model = Build_Model(x_train_fold ,pSessionParameters , pTrainingParameters  )  # Session and training parameters to define the type of model        

        # Check if LoadWeights is part of the ModelBuildParameters
        if 'LoadWeights' in pSessionParameters['ModelBuildParameters'].keys():
            # Check if Weights will be loaded for this model
            if pSessionParameters['ModelBuildParameters']['LoadWeights']:
                # Check if this fold has a weight file information
                if WeightsFileName[i]!='' and WeightsFileGoogleDriveID[i]!='':
                    print ('Loading model weights : '+ WeightsFileName[i]) 
                    if gLoadFilesFromGoogleDriveFlag: 
                        DownloadFileFromGoogleDrive(WeightsFileName[i], WeightsFileGoogleDriveID[i])
                    model.load_weights(WeightsFileName[i])
                    print ('Successfuly Loaded model weights : '+ WeightsFileName[i])        
        
      elif TrainingMode == "RESUME" or TrainingMode == "RESUME_FINETUNE" :
        # Use previous training model passed as parameter
        model = pTrainingSessionResults['models_all'][i]
        print ('Using existing training model...')
      elif TrainingMode == "FINETUNE" :   
        # Update previous training model passed which is passed as a parameter but update the model for Finetune by freezing some layers and setting others as trainable
        model = Build_Model_Finetune( pTrainingSessionResults['models_all'][i], pSessionParameters['Finetune_Trainable_Layers']  ) 
        print ('Updating existing mode for finetuning...')
      else:
        model = None
        print ('ERROR in training mode parameter')
        # raise exception  here
        
      
      model.compile(loss=pTrainingParameters['Loss'] ,optimizer=optimizers.Adam(lr=pTrainingParameters['LR']),metrics=['acc'])
      
      if i==0 :    
          print('*** MODEL INFO***')
          print('Number of layers: '+str(len(model.layers)) )
          print('IncludeTop: '+ str(pSessionParameters['ModelBuildParameters']['IncludeTop']) )                    
          print('Number of Classes: '+str(pSessionParameters['ModelBuildParameters']['NoClasses']) )
          print('Activation: '+pSessionParameters['ModelBuildParameters']['Activation'] )          
          model.summary()

          if 'Model' in str(type(model.layers[0])):
              print('')
              print('*** LAYER[0] MODEL INFO***')
              model.layers[0].summary()
          
      
     
      # Check if we will build a new data generator or continue training of a previous model which already has the data generators
      if TrainingMode in ["NEW","FINETUNE"]:             
        # New training or Finetuning session. 
         if AugFlag: 
          datagen = ImageDataGenerator(
            featurewise_center = pAugParameters['featurewise_center'],  # set input mean to 0 over the dataset
            samplewise_center = pAugParameters['samplewise_center'],  # set each sample mean to 0
            featurewise_std_normalization = pAugParameters['featurewise_std_normalization'],  # divide inputs by std of the dataset
            samplewise_std_normalization = pAugParameters['samplewise_std_normalization'],  # divide each input by its std
            zca_whitening = pAugParameters['zca_whitening'],  # apply ZCA whitening
            rotation_range = pAugParameters['rotation_range'],  # randomly rotate  in the range (degrees, 0 to 180)
            width_shift_range = pAugParameters['width_shift_range'],  # randomly shift  horizontally (fraction of total width)
            height_shift_range = pAugParameters['height_shift_range'],  # randomly shift vertically (fraction of total height)
            shear_range = pAugParameters['shear_range'],
            zoom_range = pAugParameters['zoom_range'],
            horizontal_flip = pAugParameters['horizontal_flip'],  # randomly flip 
            vertical_flip = pAugParameters['vertical_flip'], # randomly flip 
            fill_mode = pAugParameters['fill_mode'])          
          datagen.fit(x_train_fold)
        # End if pAugFlag:
      elif TrainingMode in ["RESUME", "RESUME_FINETUNE"]:
        if AugFlag:
          datagen = pTrainingSessionResults['datagen_all'][i]
      # End if pTrainingMode == "NEW": 

      
      batch_size = pTrainingParameters['BatchSize']
      epochs = pTrainingParameters['NoEpochs']
      verbose = pTrainingParameters['Verbose']
      
      
      # *** CALLBACKS ***
      lr_scheduler = LearningRateScheduler(Build_Model_LRSchedule , verbose=1)
      # Get LR Schedule params
      if 'LR_Reducer' in pTrainingParameters.keys():
          LR_Reducer_Param = pTrainingParameters['LR_Reducer']
          lr_reducer = ReduceLROnPlateau(monitor=LR_Reducer_Param['Monitor'], factor=LR_Reducer_Param['Factor'], cooldown=LR_Reducer_Param['Cooldown'], patience=LR_Reducer_Param['Patience'], min_lr=LR_Reducer_Param['Min_lr'], min_delta=LR_Reducer_Param['Min_delta'], verbose=1)
      else:
          lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, cooldown=0, patience=50, min_lr=1e-5, min_delta=0.001, verbose=1)
          
         
      callbacks = [lr_reducer]            
      
      print ('')
    
      # Perform Training  
      if AugFlag:  
        # Fit the model on the batches generated by datagen.flow().        
        print ('Performing training WITH DATA AUGMENTATION using fit_generator() ...')
        history = model.fit_generator(datagen.flow(x_train_fold,y_train_fold,batch_size=batch_size),
                                steps_per_epoch=x_train_fold.shape[0] // batch_size,
                                epochs=epochs, verbose=verbose,callbacks=callbacks,
                                validation_data=(x_val_fold, y_val_fold) )
      else:
        print ('Performing normal training WITHOUT DATA AUGMENTATION using fit() ...')        
        history = model.fit(x_train_fold,y_train_fold,batch_size=batch_size,epochs=epochs,verbose=verbose,callbacks=callbacks,validation_data=(x_val_fold, y_val_fold) )
           
      print ('')
      all_history.append(history.history) 

      # Evaluate the model on the training, validation and test data
      scores = dict()
      
      print ('Training Scores:')
      score_train = model.evaluate(x_train_fold, y_train_fold, verbose=0)
      print (score_train)
      scores['train'] = score_train

      print ('Validation Scores:')
      score_val = model.evaluate(x_val_fold, y_val_fold, verbose=0)
      print (score_val)
      scores['val'] = score_val

      print ('Testing Scores:')
      score_test = model.evaluate(pDataFoldsAll['x_test'], pDataFoldsAll['y_test'], verbose=0)
      print (score_test)
      scores['test'] = score_test

      # Append the training, val and test scores of this fold to the all_scores list
      all_scores.append(scores)
            
      all_models.append(model)      
      if AugFlag:        
        all_datagen.append(datagen)        
      
      print ('')

      print ('*** Predictions and ROC ***')
      # Updates for Calculating Predictions and ROC curve
      predictions = dict()
      predictions['train'] = dict()
      predictions['val'] = dict()
      predictions['test'] = dict()
      
      # *** Predictions for Training Data ***
      y_predictions = model.predict(x_train_fold)                # y output predictions for current fold 
      class_predictions = y_predictions.argmax(axis=-1)     # Class predictions for current fold 

      # Check if Label data is Categorical for SOFTMAX activation
      if pSessionParameters['ModelBuildParameters']['Activation'] == 'softmax':
          ROC_FPR, ROC_TPR, ROC_thresholds = roc_curve(y_train_fold.ravel(), y_predictions.ravel())  # ROC curve FPR, TPR and thresholds          
      else:
          ROC_FPR, ROC_TPR, ROC_thresholds = roc_curve(y_train_fold, y_predictions)  # ROC curve FPR, TPR and thresholds
      ROC_AUC =  auc(ROC_FPR, ROC_TPR)             # Area under ROC curve 
      print('Training ROC AUC: %0.2f' % ROC_AUC )    
      # Add predictions and ROC data for this fold to the predictions dict      
      predictions['train']['y_predictions'] = y_predictions
      predictions['train']['class_predictions'] = class_predictions
      predictions['train']['ROC_FPR'] = ROC_FPR 
      predictions['train']['ROC_TPR'] = ROC_TPR         
      predictions['train']['ROC_thresholds'] = ROC_thresholds
      predictions['train']['ROC_AUC'] = ROC_AUC         
      
      # *** Predictions for Validation Data ***
      y_predictions = model.predict(x_val_fold)                # y output predictions for current fold 
      class_predictions = y_predictions.argmax(axis=-1)     # Class predictions for current fold       

      # Check if Label data is Categorical for SOFTMAX activation
      if pSessionParameters['ModelBuildParameters']['Activation'] == 'softmax':
          ROC_FPR, ROC_TPR, ROC_thresholds = roc_curve(y_val_fold.ravel(), y_predictions.ravel())  # ROC curve FPR, TPR and thresholds          
      else:
          ROC_FPR, ROC_TPR, ROC_thresholds = roc_curve(y_val_fold, y_predictions)  # ROC curve FPR, TPR and thresholds

      ROC_AUC =  auc(ROC_FPR, ROC_TPR)            # Area under ROC curve 
      print('Validation ROC AUC: %0.2f' % ROC_AUC )
      # Add predictions and ROC data for this fold to the predictions dict      
      predictions['val']['y_predictions'] = y_predictions
      predictions['val']['class_predictions'] = class_predictions
      predictions['val']['ROC_FPR'] = ROC_FPR 
      predictions['val']['ROC_TPR'] = ROC_TPR         
      predictions['val']['ROC_thresholds'] = ROC_thresholds
      predictions['val']['ROC_AUC'] = ROC_AUC         
      
      # *** Predictions for Test Data ***
      y_predictions = model.predict(pDataFoldsAll['x_test'])                # y output predictions for current fold 
      class_predictions = y_predictions.argmax(axis=-1)     # Class predictions for test labels             

      # Check if Label data is Categorical for SOFTMAX activation
      if pSessionParameters['ModelBuildParameters']['Activation'] == 'softmax':
          ROC_FPR, ROC_TPR, ROC_thresholds = roc_curve(pDataFoldsAll['y_test'].ravel(), y_predictions.ravel())  # ROC curve FPR, TPR and thresholds          
      else:
          ROC_FPR, ROC_TPR, ROC_thresholds = roc_curve(pDataFoldsAll['y_test'], y_predictions)  # ROC curve FPR, TPR and thresholds

      ROC_AUC =  auc(ROC_FPR, ROC_TPR)            # Area under ROC curve 
      print('Test ROC AUC: %0.2f' % ROC_AUC )
      # Add predictions and ROC data for this fold to the predictions dict 
      predictions['test']['y_predictions'] = y_predictions
      predictions['test']['class_predictions'] = class_predictions
      predictions['test']['ROC_FPR'] = ROC_FPR 
      predictions['test']['ROC_TPR'] = ROC_TPR         
      predictions['test']['ROC_thresholds'] = ROC_thresholds
      predictions['test']['ROC_AUC'] = ROC_AUC      
      
      # Append predictions and ROC data for this fold to the all_predictions list      
      all_predictions.append(predictions)  
      
      # Add an empty SessionResultsInfo dict which will be updated manualy after the results are evaluated 
      resultsinfo = dict()    
      resultsinfo['TrainingGroupFlag'] = '' 
      resultsinfo['TrainingGroupWeightsFlag'] = ''
      resultsinfo['TrainingGroupModelFlag'] = ''      
      resultsinfo['TrainingGroupBestSessionID'] = ''            
      resultsinfo['TrainingGroupBestSessionFlag'] = ''                  
      resultsinfo['SessionWeightsFlag'] = '' 
      resultsinfo['SessionModelFlag'] = '' 
      resultsinfo['WeightsFileName'] = '' 
      resultsinfo['WeightsFileGoogleID'] = ''         
      resultsinfo['ModelFileName'] = '' 
      resultsinfo['ModelFileGoogleID'] = ''               
      all_resultsinfo.append(resultsinfo) 
      
      # end for i  (Loop for all training folds)

      # Return all the training results in a single dict to make the code simpler
      TrainingSessionResults = dict()   
      TrainingSessionResults['models_all'] = all_models      
      TrainingSessionResults['datagen_all'] = all_datagen     
      TrainingSessionResults['history_all'] = all_history    
      TrainingSessionResults['scores_all'] = all_scores  
      TrainingSessionResults['predictions_all'] = all_predictions        
      TrainingSessionResults['resultsinfo_all'] = all_resultsinfo
      
  return TrainingSessionResults




# *** TrainCrossValidation_Fold ***

def TrainCrossValidation_Fold( px_train_conv_aug, px_test_conv_aug, py_train_aug, py_test_aug, pSessionParameters , pTrainingParameters, pAugParameters , pFoldIndex=0, pTrainingSessionResults=None):  
  ''' 
  # Performs training using cross validation on ONLY A SINGLE fold
 
  Input parameters:
    pDataFoldsAll: dict with all Train, Val datafolds as lists and Test data as numpy arrays
                   {x_train_folds_all, x_val_folds_all, y_train_folds_all, y_val_folds_all, x_test, y_test}
    px_test_conv_aug: Test data  (included in the DataFolds dict)
    py_test_aug: Test labels (included in the DataFolds dict)
    pNoFolds: number of training folds
    pSessionParameters: dict with session parameters. 
              SessionParameters = dict(TrainingMode='NEW',       # 1) NEW 2) RESUME 3) FINETUNE  4) RESUME_FINETUNE
                                       Finetune_Trainable_Layers=[]    # Layers to be set as trainable during finetuning
                                       ModelBuildParameters =''   # params to define how the model layers will be built 
                                       BatchNorm = 'NONE'   # Flag to indocate whether or not to add batch norm in the model
                                                              1) NONE 2) CONV_ONLY 3) CONV_DENSE (Both Conv and Dense layers)
    ***                               )     
    TrainingMode:  Text indicating training mode:  1) NEW 2) RESUME 3) FINETUNE  4) RESUME_FINETUNE
                   Text Flag to indicate whether to create new models or use previous models which were trained already and available in the pTrainingSessionResults 
                   This is needed to continue more training epochs from previous training sessions.
                   1) New: A new training model will be created and a new trainig session is executed (Similar to previous pInitModelsFlag=True)
                   2) Resume: This will resume the previous trainings using the models and dataaug data passed on from previous sessions in the pTrainingSessionResults
                   3) Finetune: This will finetune the existing models and fine tune the existing training sessions. The previous models are needed which will be obtained 
                                also from the parameter pTrainingSessionResults.
                   4) Resume_Finetune: This will resume the finetunine of a previous finetune sessions. 
    ***
    
    pTrainingParameters: dict with training parameters: Example
          TrainingParameters = dict(
                                NoEpochs=100,  # Number of training epochs
                                BatchSize= 10  # training batch size
                                Loss='binary_crossentropy'
                                Optimizers='Adam'
                                LR =1e-4
                                Metrics='acc',
                                Verbose=1)
    pAugFlag: To indicate if data augmentation will be used
    pAugParameters: dict with augmentation parameters: Example
          AugParameters = dict(
                                featurewise_center=False,  # set input mean to 0 over the dataset
                                samplewise_center=False,  # set each sample mean to 0
                                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                                samplewise_std_normalization=False,  # divide each input by its std
                                zca_whitening=False,  # apply ZCA whitening
                                rotation_range=5,  # randomly rotate in the range (degrees, 0 to 180)
                                width_shift_range=0.05,  # randomly shift horizontally (fraction of total width)
                                height_shift_range=0.05,  # randomly shift vertically (fraction of total height)
                                shear_range=0.05,
                                zoom_range=0,
                                horizontal_flip=True,  # randomly flip 
                                vertical_flip=False, # randomly flip 
                                fill_mode='nearest')   
    pTrainingSessionResults: All training results and structures from previous training to be used in the mode of resuming a previous training.
                            It contains all the previous models and data generators in addition to the training scores and history.

    
     
    *** NEW TRAINING MODES ALL UNDER: NEW ***
    1) NEW   2) NEW -> RESUME   3) NEW -> FINETUNE
    - Summary of steps:
        1) NEW training session with a selected Model Type which has to be identical to the Model used to create the weights.
        2) Build a new model based on Model Type but without the classifier Top which is always the default anyway.
        3) (OPTIONAL) Add Dense and/or Dropout Layers using the ModelBuildParameters: DenseLayers, DropoutLayers. 
        4) Add a classifer top based on the no of classes and type of activation.
        5) (OPTIONAL) LOAD WEIGHTS for all folds from the previous trained models using the WeightFiles in the ModelBuildParameters
           This is optional for NEW training but MANDATORY for Finetuning and Transfer Learning. This could also be used
           for RESUME function, by creating a new model, loading weights and RESUME training normally.
        6) (OPTIONAL) Configure the Model for Finetuning using the FN Build_Model_Finetune which will use the 
           Finetune_Trainable_Layers value in the session parameters. This parameter determines whether we will perform Finetuning
           This is optional for NEW training and RESUME but MANDATORY for Finetuning
        7) Train with or without Data Aug.   
        
    1) NEW:             Steps 1,2,4,7
    2) NEW -> RESUME:   Steps 1,2,4,5,7
    3) NEW -> FINETUNE: Steps 1,2,3,4,5,6,7

  Return parameters: 
       TrainingSessionResults: dict with the following lists:
                               all_models: all trained models for all data folds
                               all_history: training history of all data folds
                               all_scores_training: evaluation of training scores for all folds 
                               all_scores_validation: evaluation of validation scores for all folds          
                               all_models: all_models (to be used to continue training session if needed)
                               all_datagen: all data generators (to be used to continue training session if needed)
       
  '''
  # One Loop for this training fold
  i = pFoldIndex      
  
  # Create lists if this is the first fold  
  #if i==0:  
  if pTrainingSessionResults==None:
      all_scores = []
      all_history = []
      all_models = []
      all_datagen = []
      all_predictions = [] 
      all_resultsinfo = []
  else:
      # Get the lists from the existing TrainingSessionResults
      all_models = pTrainingSessionResults['models_all']       
      all_datagen = pTrainingSessionResults['datagen_all']      
      all_history = pTrainingSessionResults['history_all']  
      all_scores = pTrainingSessionResults['scores_all']  
      all_predictions = pTrainingSessionResults['predictions_all']          
      all_resultsinfo = pTrainingSessionResults['resultsinfo_all']  


  # Get the Data for the Fold
  DataFold = BuildCrossValidationData_Fold(pSessionParameters, px_train_conv_aug, px_test_conv_aug, py_train_aug, py_test_aug, pFoldIndex )

  # Get Training and Validation data folds from the input parameter dict
  all_x_train_folds = DataFold['x_train_folds_all']   # List with all training data folds 
  all_x_val_folds = DataFold['x_val_folds_all']       # List with all validation data folds 
  all_y_train_folds = DataFold['y_train_folds_all']   # List with all training label folds 
  all_y_val_folds = DataFold['y_val_folds_all']       # List with all validation label folds 
   
  # Get lists which has Weights FileNames and GoogleDriveIDs if the Weights will be loaded
  # Check if LoadWeights is part of the ModelBuildParameters 
  if 'LoadWeights' in pSessionParameters['ModelBuildParameters'].keys():
      if pSessionParameters['ModelBuildParameters']['LoadWeights']:
          # Get the list which has the Weight file information to be loaded
          WeightsFileName = pSessionParameters['ModelBuildParameters']['WeightsFileName']   # List of FileNames for each fold
          WeightsFileGoogleDriveID = pSessionParameters['ModelBuildParameters']['WeightsFileGoogleDriveID'] # List of GoogleDriveIDs for each fold          
  
      
  print('\n'+'******************************************'+'\n')
  print(psutil.virtual_memory())
  print('Training fold # ', i+1)    

  # Get the training and validation data for fold k from the fold lists
  x_train_fold = all_x_train_folds[0]
  x_val_fold = all_x_val_folds[0]
  y_train_fold = all_y_train_folds[0]
  y_val_fold = all_y_val_folds[0]



  print ('Training input shape ' + str(x_train_fold.shape))
  print ('Training target shape ' + str(y_train_fold.shape))
  print ('Val input shape ' + str(x_val_fold.shape))
  print ('Val target shape ' + str(y_val_fold.shape))

  print ('')

  ' Get session parameters for training mode and data augmentation'
  TrainingMode = pSessionParameters['TrainingMode']
  AugFlag = pSessionParameters['DataAug']
  

  # Check if we will build a new model or continue training of a previous model
   if TrainingMode == "NEW":     
    # Create a new model
    print ('New training model created...')
    model = Build_Model(x_train_fold ,pSessionParameters , pTrainingParameters  )  # Session and training parameters to define the type of model        

    # Check if LoadWeights is part of the ModelBuildParameters for backward compatability
    if 'LoadWeights' in pSessionParameters['ModelBuildParameters'].keys():
        # Check if Weights will be loaded for this model
        if pSessionParameters['ModelBuildParameters']['LoadWeights']:
            # Check if this fold has a weight file information
            if WeightsFileName[i]!='' and WeightsFileGoogleDriveID[i]!='':
                print ('Loading model weights : '+ WeightsFileName[i]) 
                if gLoadFilesFromGoogleDriveFlag: 
                    DownloadFileFromGoogleDrive(WeightsFileName[i], WeightsFileGoogleDriveID[i])
                model.load_weights(WeightsFileName[i])
                print ('Successfuly Loaded model weights : '+ WeightsFileName[i])        
                
    # Check if Finetune_Trainable_Layers is not empty which means we will perform TRANSFER LEARNING and FINETUNING
    #Update previous training model passed which is passed as a parameter but update the model for Finetune by freezing some layers and setting others as trainable
    if len(pSessionParameters['Finetune_Trainable_Layers'])!=0:
        print ('Updating existing mode for Transfer Learning and Finetuning...')                
        model = Build_Model_Finetune( model, pSessionParameters  )  

    # Check if Finetune_Start_BlockPrefixF is not empty which means we will perform TRANSFER LEARNING and FINETUNING using GeneXNet
    if 'Finetune_Start_BlockPrefix' in pSessionParameters.keys():
        if pSessionParameters['Finetune_Start_BlockPrefix']!='':            
            print ('Updating existing model for GeneXNet Transfer Learning and Finetuning...')                
            model = Build_Model_Finetune_GeneXNet( model, pSessionParameters  )  
    
   
  elif TrainingMode == "RESUME" or TrainingMode == "RESUME_FINETUNE" :
    # Use previous training model passed as parameter
    model = pTrainingSessionResults['models_all'][i]
    print ('Using existing training model...')
  elif TrainingMode == "FINETUNE" :   
    # Update previous training model passed which is passed as a parameter but update the model for Finetune by freezing some layers and setting others as trainable
    model = Build_Model_Finetune( pTrainingSessionResults['models_all'][i], pSessionParameters['Finetune_Trainable_Layers']  )  # 2nd parameter to determine trainable layers
    print ('Updating existing mode for finetuning...')
  else:
    model = None
    print ('ERROR in training mode parameter')
 
  
  model.compile(loss=pTrainingParameters['Loss'] ,optimizer=optimizers.RMSprop(lr=pTrainingParameters['LR']),metrics=['acc'])
  
  if pTrainingSessionResults==None:   # First Fold Iteration so show model info
      print('*** MODEL INFO***')
      print('Number of layers: '+str(len(model.layers)) )
      print('IncludeTop: '+ str(pSessionParameters['ModelBuildParameters']['IncludeTop']) )                    
      print('Number of Classes: '+str(pSessionParameters['ModelBuildParameters']['NoClasses']) )
      print('Activation: '+pSessionParameters['ModelBuildParameters']['Activation'] )          
      model.summary()
      if 'Model' in str(type(model.layers[0])):
          print('')
          print('*** LAYER[0] MODEL INFO***')
          model.layers[0].summary()
      
 
  # Check if we will build a new data generator or continue training of a previous model which already has the data generators
  if TrainingMode in ["NEW","FINETUNE"]:             
    # New training or Finetuning session. 
     if AugFlag: 
      datagen = ImageDataGenerator(
        featurewise_center = pAugParameters['featurewise_center'],  # set input mean to 0 over the dataset
        samplewise_center = pAugParameters['samplewise_center'],  # set each sample mean to 0
        featurewise_std_normalization = pAugParameters['featurewise_std_normalization'],  # divide inputs by std of the dataset
        samplewise_std_normalization = pAugParameters['samplewise_std_normalization'],  # divide each input by its std
        zca_whitening = pAugParameters['zca_whitening'],  # apply ZCA whitening
        rotation_range = pAugParameters['rotation_range'],  # randomly rotate in the range (degrees, 0 to 180)
        width_shift_range = pAugParameters['width_shift_range'],  # randomly shift horizontally (fraction of total width)
        height_shift_range = pAugParameters['height_shift_range'],  # randomly shift vertically (fraction of total height)
        shear_range = pAugParameters['shear_range'],
        zoom_range = pAugParameters['zoom_range'],
        horizontal_flip = pAugParameters['horizontal_flip'],  # randomly flip 
        vertical_flip = pAugParameters['vertical_flip'], # randomly flip
        fill_mode = pAugParameters['fill_mode'])          
      datagen.fit(x_train_fold)
    # End if pAugFlag:
  elif TrainingMode in ["RESUME", "RESUME_FINETUNE"]:
    if AugFlag:
      datagen = pTrainingSessionResults['datagen_all'][i]
  
  batch_size = pTrainingParameters['BatchSize']
  epochs = pTrainingParameters['NoEpochs']
  verbose = pTrainingParameters['Verbose']
  
  
  # *** CALLBACKS ***
  lr_scheduler = LearningRateScheduler(Build_Model_LRSchedule , verbose=1)
  # Get LR Schedule params
  if 'LR_Reducer' in pTrainingParameters.keys():
      LR_Reducer_Param = pTrainingParameters['LR_Reducer']
      lr_reducer = ReduceLROnPlateau(monitor=LR_Reducer_Param['Monitor'], factor=LR_Reducer_Param['Factor'], cooldown=LR_Reducer_Param['Cooldown'], patience=LR_Reducer_Param['Patience'], min_lr=LR_Reducer_Param['Min_lr'], min_delta=LR_Reducer_Param['Min_delta'], verbose=1)
  else:
      lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, cooldown=0, patience=50, min_lr=1e-5, min_delta=0.001, verbose=1)
      
 

  if 'EarlyStopping' in pTrainingParameters.keys():
      EarlyStopping_Param = pTrainingParameters['EarlyStopping']
      EarlyStoppingCallBack = EarlyStopping(monitor=EarlyStopping_Param['Monitor'], min_delta=EarlyStopping_Param['Min_delta'], patience=EarlyStopping_Param['Patience'],baseline=EarlyStopping_Param['baseline'],restore_best_weights=EarlyStopping_Param['restore_best_weights'],  verbose=EarlyStopping_Param['verbose'])
  else:
      EarlyStoppingCallBack = EarlyStopping(monitor='val_acc', min_delta=0.0001,  patience=epochs,baseline=None , restore_best_weights=False, verbose=1)          

      
  callbacks = [lr_reducer, EarlyStoppingCallBack]            
  
  print ('')

  if AugFlag:  
    # Fit the model on the batches generated by datagen.flow().        
    print ('Performing training WITH DATA AUGMENTATION using fit_generator() ...')
    history = model.fit_generator(datagen.flow(x_train_fold,y_train_fold,batch_size=batch_size),
                            steps_per_epoch=x_train_fold.shape[0] // batch_size,
                            epochs=epochs, verbose=verbose,callbacks=callbacks,
                            validation_data=(x_val_fold, y_val_fold) )
  else:
    print ('Performing normal training WITHOUT DATA AUGMENTATION using fit() ...')        
    history = model.fit(x_train_fold,y_train_fold,batch_size=batch_size,epochs=epochs,verbose=verbose,callbacks=callbacks,validation_data=(x_val_fold, y_val_fold) )
       
  print ('')
  all_history.append(history.history) 

  # Evaluate the model on the training, validation and test data
  scores = dict()
  
  print ('Training Scores:')
  score_train = model.evaluate(x_train_fold, y_train_fold, verbose=0)
  print (score_train)
  scores['train'] = score_train

  print ('Validation Scores:')
  score_val = model.evaluate(x_val_fold, y_val_fold, verbose=0)
  print (score_val)
  scores['val'] = score_val

  print ('Testing Scores:')
  score_test = model.evaluate(DataFold['x_test'], DataFold['y_test'], verbose=0)
  print (score_test)
  scores['test'] = score_test

  # Append the training, val and test scores of this fold to the all_scores list
  all_scores.append(scores)
        
  all_models.append(model)      

  if AugFlag:        
    all_datagen.append(datagen)        
  
  print ('')

  print ('*** Predictions and ROC ***')
  # Updates for Calculating Predictions and ROC curve
  predictions = dict()
  predictions['train'] = dict()
  predictions['val'] = dict()
  predictions['test'] = dict()
  
  # *** Predictions for Training Data ***
  y_predictions = model.predict(x_train_fold)                # y output predictions for current fold 
  class_predictions = y_predictions.argmax(axis=-1)     # Class predictions for current fold 
  # Check if Label data is Categorical for SOFTMAX activation
  if pSessionParameters['ModelBuildParameters']['Activation'] == 'softmax':
      ROC_FPR, ROC_TPR, ROC_thresholds = roc_curve(y_train_fold.ravel(), y_predictions.ravel())  # ROC curve FPR, TPR and thresholds          
  else:
      ROC_FPR, ROC_TPR, ROC_thresholds = roc_curve(y_train_fold, y_predictions)  # ROC curve FPR, TPR and thresholds
  ROC_AUC =  auc(ROC_FPR, ROC_TPR)             # Area under ROC curve 
  print('Training ROC AUC: %0.2f' % ROC_AUC )    
  # Add predictions and ROC data for this fold to the predictions dict      
  predictions['train']['y_predictions'] = y_predictions
  predictions['train']['class_predictions'] = class_predictions
  predictions['train']['ROC_FPR'] = ROC_FPR 
  predictions['train']['ROC_TPR'] = ROC_TPR         
  predictions['train']['ROC_thresholds'] = ROC_thresholds
  predictions['train']['ROC_AUC'] = ROC_AUC         
  
  # *** Predictions for Validation Data ***
  y_predictions = model.predict(x_val_fold)                # y output predictions for current fold 
  class_predictions = y_predictions.argmax(axis=-1)     # Class predictions for current fold       

  # Check if Label data is Categorical for SOFTMAX activation
  if pSessionParameters['ModelBuildParameters']['Activation'] == 'softmax':
      ROC_FPR, ROC_TPR, ROC_thresholds = roc_curve(y_val_fold.ravel(), y_predictions.ravel())  # ROC curve FPR, TPR and thresholds          
  else:
      ROC_FPR, ROC_TPR, ROC_thresholds = roc_curve(y_val_fold, y_predictions)  # ROC curve FPR, TPR and thresholds

  ROC_AUC =  auc(ROC_FPR, ROC_TPR)            # Area under ROC curve 
  print('Validation ROC AUC: %0.2f' % ROC_AUC )
  # Add predictions and ROC data for this fold to the predictions dict      
  predictions['val']['y_predictions'] = y_predictions
  predictions['val']['class_predictions'] = class_predictions
  predictions['val']['ROC_FPR'] = ROC_FPR 
  predictions['val']['ROC_TPR'] = ROC_TPR         
  predictions['val']['ROC_thresholds'] = ROC_thresholds
  predictions['val']['ROC_AUC'] = ROC_AUC         
  
  # *** Predictions for Test Data ***
  y_predictions = model.predict(DataFold['x_test'])                # y output predictions for current fold 
  class_predictions = y_predictions.argmax(axis=-1)     # Class predictions for test labels             

  # Check if Label data is Categorical for SOFTMAX activation
  if pSessionParameters['ModelBuildParameters']['Activation'] == 'softmax':
      ROC_FPR, ROC_TPR, ROC_thresholds = roc_curve(DataFold['y_test'].ravel(), y_predictions.ravel())  # ROC curve FPR, TPR and thresholds          
  else:
      ROC_FPR, ROC_TPR, ROC_thresholds = roc_curve(DataFold['y_test'], y_predictions)  # ROC curve FPR, TPR and thresholds

  ROC_AUC =  auc(ROC_FPR, ROC_TPR)            # Area under ROC curve 
  print('Test ROC AUC: %0.2f' % ROC_AUC )
  # Add predictions and ROC data for this fold to the predictions dict 
  predictions['test']['y_predictions'] = y_predictions
  predictions['test']['class_predictions'] = class_predictions
  predictions['test']['ROC_FPR'] = ROC_FPR 
  predictions['test']['ROC_TPR'] = ROC_TPR         
  predictions['test']['ROC_thresholds'] = ROC_thresholds
  predictions['test']['ROC_AUC'] = ROC_AUC      
  
  # Append predictions and ROC data for this fold to the all_predictions list      
  all_predictions.append(predictions)  
  
  # Add an empty SessionResultsInfo dict which will be updated manualy after the results are evaluated 
  resultsinfo = dict()    
  resultsinfo['TrainingGroupFlag'] = '' 
  resultsinfo['TrainingGroupWeightsFlag'] = ''
  resultsinfo['TrainingGroupModelFlag'] = ''      
  resultsinfo['TrainingGroupBestSessionID'] = ''            
  resultsinfo['TrainingGroupBestSessionFlag'] = ''                  
  resultsinfo['SessionWeightsFlag'] = '' 
  resultsinfo['SessionModelFlag'] = '' 
  resultsinfo['WeightsFileName'] = '' 
  resultsinfo['WeightsFileGoogleID'] = ''         
  resultsinfo['ModelFileName'] = '' 
  resultsinfo['ModelFileGoogleID'] = ''               
  all_resultsinfo.append(resultsinfo) 
      
      

  # end for i  (Loop for all training folds)

  # Create the TrainingSessionResults if this is the first fold  
  if pTrainingSessionResults==None:      
      # Return all the training results in a single dict to make the code simpler
      TrainingSessionResults = dict()   
      TrainingSessionResults['models_all'] = all_models      
      TrainingSessionResults['datagen_all'] = all_datagen     
      TrainingSessionResults['history_all'] = all_history    
      TrainingSessionResults['scores_all'] = all_scores  
      TrainingSessionResults['predictions_all'] = all_predictions        
      TrainingSessionResults['resultsinfo_all'] = all_resultsinfo
  else:
      # Get the training results from the passed parameter which has the information for this fold and all previous folds
      TrainingSessionResults = pTrainingSessionResults
   
  del DataFold
  
  return TrainingSessionResults


# **** TrainCrossValidation_AllFolds ***
def TrainCrossValidation_AllFolds( px_train_conv_aug, px_test_conv_aug, py_train_aug, py_test_aug, pSessionParameters , pTrainingParameters, pAugParameters, pBatchNo=999,pBatchSerialCount=9  ):  
  ''' 
  # Performs training using cross validation for all the Folds but one at a time. 
  # This is needed to optimize on memory usage and avoid running out of memory for large datasets                      
  # The fn will first call TrainCrossValidation_Fold and BuildCrossValidationData_Fold to get each Fold during each iteration 
    and perform the training for each fold
       
  '''
  global gUploadFilesToGoogleDriveFlag
  
  if 'NoFoldIterations' in pTrainingParameters.keys():
      k = pTrainingParameters['NoFoldIterations']
  else:
      k = pSessionParameters['NoFolds'] 
  

  if 'FoldIndexShift' in pTrainingParameters.keys():
      FoldIndexShift = pTrainingParameters['FoldIndexShift']
  else:
      FoldIndexShift = 0  # Normal 



  for i in range(k):
      if i==0:          
          TrainingSessionResults = TrainCrossValidation_Fold( px_train_conv_aug, px_test_conv_aug, py_train_aug, py_test_aug, pSessionParameters , pTrainingParameters, pAugParameters , pFoldIndex = i + FoldIndexShift)
      else:
          TrainingSessionResults = TrainCrossValidation_Fold( px_train_conv_aug, px_test_conv_aug, py_train_aug, py_test_aug, pSessionParameters , pTrainingParameters, pAugParameters , pFoldIndex = i + FoldIndexShift, pTrainingSessionResults = TrainingSessionResults)            
         
      # Save Intermediate Fold Iterations SessionData and Weights
      if 'SaveIntermediateFoldIterationsFlag' in pTrainingParameters.keys():
          SaveIntermediateFoldIterationsFlag = pTrainingParameters['SaveIntermediateFoldIterationsFlag']
      else:
          SaveIntermediateFoldIterationsFlag = False  # Default to Save
          
      if SaveIntermediateFoldIterationsFlag:          
          print('')
          print ('*** Saving Intermediate session data *** Fold Iteration: '+str(i+1))
    
          # init GoogleDrive Auth Again because the session might be lost after a long training activity
          if gUploadFilesToGoogleDriveFlag:
              InitGoogleDriveAuth()

          # Create a TEMP intermediate SessionDataArchiveAll to hold intermediate results
          SessionDataArchiveAll_Intermediate =[]  # It has to be rest every iteration so that it is accumulative
          
          TrainingParameters_Intermediate = copy.deepcopy(pTrainingParameters)        
          TrainingParameters_Intermediate['NoFoldIterations'] = i+1
          
          # Update Evaluation
          eval='PIPELINE'
          pSessionParameters['Evaluation'] = eval
          pSessionParameters['BestFold'] = 1   # Init with any value and will be adjusted during training 
    
          # Update Session Data        
          SessionDataArchiveAll_Intermediate = UpdateSessionData(SessionDataArchiveAll_Intermediate, pSessionParameters ,TrainingParameters_Intermediate,pAugParameters, TrainingSessionResults)
          print('Sessions Saved successfuly. Total sessions: '+str(len(SessionDataArchiveAll_Intermediate)))  
           
          # Save Session Weights for current Fold ONLY
          print ('Saving Intermediate session weights...')
          SetPipelineBatchNo( pBatchNo+'_{:0>2d}'.format(pBatchSerialCount)+'_{:0>2d}'.format(i+1) )    
          # The Save weights will be filtered for this Fold only so that the weights dont get saved multiple times for each fold iteration
          SaveSessionWeights(TrainingSessionResults, SessionDataArchiveAll_Intermediate, True, [i+1]) 
          print ('Saving Intermediate session data...')
          FileName, FileGoogleID = SaveSessionData(SessionDataArchiveAll_Intermediate)
          print(FileName)        
          SetPipelineBatchNo( pBatchNo) 
          # End if Save Intermediate Fold Iterations         

  return TrainingSessionResults




'''
******************************************************************************
*** Module: GeneXNet (Gene eXpression Network) ***

*** SESSION INPUT/OUTPUT MANAGEMENT ***

******************************************************************************
'''


# **** UpdateSessionData ***
def UpdateSessionData(pSessionDataAll, pSessionParameters ,pTrainingParameters,pAugParameters, pTrainingSessionResults):
  '''
  - Inserts all the training session data into a list of dictionaries so it can be saved to a json file. 

    *** Input parameters: ***
    pSessionDataAll: List of dict which contains the history of Training Sessions that have been previously recorded to be saved in a json file
    pTrainingSessionResults: All training results and structures from last training. Contains all models,data generators, training scores and history.
   
  '''
  
  VersionNo=1.5
  
  # Check if there is already session data from previous sessions
  if len(pSessionDataAll)==0:
    # Create an empty list for the session data
    session_data_all = []
    FoldGroupSessionNo = 1   # Init Fold Group SessionNo (This was passed manually before but now it will be calculated automatically)
  else:
    # create a reference to the existing session data
    session_data_all = pSessionDataAll
    # Get the Fold Group SessionNo from the previous session and inc by 1
    FoldGroupSessionNo = pSessionDataAll[len(pSessionDataAll)-1]['SessionParameters']['SessionNo'] + 1
    
    
  
  # Create a unique id for this fold group which can be used to group the folds together if needed
  FoldGroupID = str(uuid.uuid4())  
  

  # if this training mode is 'NEW' then create a unique training group id which can be used to group the training sessions of new, resume and finetuning together
  if pSessionParameters['TrainingMode'] =='NEW': 
    # Create a unique id training group id which can be used to group the training sessions of new, resume and finetuning together
    TrainingGroupID = str(uuid.uuid4())      
  else:
    # Get the training group ID from the previous session which was probably NEW or RESUME or RESUME_FINETUNE
    TrainingGroupID = pSessionDataAll[len(pSessionDataAll)-1]['SessionParameters']['TrainingGroupID']
      
  # Loop for all the folds in the pTrainingSessionResults
  for i in range(len(pTrainingSessionResults['models_all'])):
      SessionData = dict()
      SessionData['SessionID'] = str(uuid.uuid4())
      TimeStamp = datetime.datetime.now()
      SessionData['SessionDateTime'] = TimeStamp.strftime("%Y-%m-%d %H:%M")    
      SessionData['TemplateVersionNo'] = VersionNo   

      # Session parameters
      SessionParameters =dict()
      # SessionParameters['SessionNo'] = pSessionParameters['SessionNo']  # This used to be passed manually but now it will be calculated
      SessionParameters['SessionNo'] = FoldGroupSessionNo
      SessionParameters['Description'] = pSessionParameters['Description'] 
      SessionParameters['Comments'] = pSessionParameters['Comments']
      SessionParameters['Evaluation'] = pSessionParameters['Evaluation']
      SessionParameters['TrainingMode'] = pSessionParameters['TrainingMode']    # 1) NEW 2) RESUME 3) FINETUNE  4) RESUME_FINETUNE
      SessionParameters['ModelBuildParameters'] = pSessionParameters['ModelBuildParameters']                  
      SessionParameters['BatchNorm'] = pSessionParameters['BatchNorm']    # 1) NONE 2) CONV_ONLY 3) CONV_DENSE (Both Conv and Dense layers)
      SessionParameters['DataAug'] = pSessionParameters['DataAug']    # Flag to indicate whether to use Data Augmentation
      SessionParameters['Finetune_Trainable_Layers'] = pSessionParameters['Finetune_Trainable_Layers']    # Layers set to trainable during finetuning                              
      SessionParameters['TrainTestSplit'] = pSessionParameters['TrainTestSplit']
      SessionParameters['CrossVal'] =  pSessionParameters['CrossVal']
      SessionParameters['NoFolds'] =  pSessionParameters['NoFolds']
      SessionParameters['BestFold'] =  pSessionParameters['BestFold']
      SessionParameters['FoldNo'] =  i+1    
      SessionParameters['FoldGroupID'] = FoldGroupID
      SessionParameters['TrainingGroupID'] = TrainingGroupID
      SessionParameters['Verbose'] =  pSessionParameters['Verbose']      

      SessionParameters['DataType'] =  pSessionParameters['DataType']      
      SessionParameters['DataFoldsSource'] =  pSessionParameters['DataFoldsSource']      
      SessionParameters['DataFoldsFileName'] =  pSessionParameters['DataFoldsFileName']                              
      SessionParameters['DataFoldsFileGoogleID'] =  pSessionParameters['DataFoldsFileGoogleID']                                    
      SessionParameters['NoClasses'] =  pSessionParameters['NoClasses']                                          
      SessionParameters['ModelType'] =  pSessionParameters['ModelType']                                          
      SessionParameters['ModelBuildFN'] =  pSessionParameters['ModelBuildFN']                                                
      SessionParameters['ModelFinetuneBuildFN'] =  pSessionParameters['ModelFinetuneBuildFN']                                                            
      SessionData['SessionParameters'] = SessionParameters

      # Training parameters
      SessionData['TrainingParameters']= pTrainingParameters

      # Augmentation parameters
      SessionData['AugParameters']= pAugParameters

      # Model used for training
      Model =dict()
      Model['Configuration'] = dict()
      Model['JSON'] =  pTrainingSessionResults['models_all'][i].to_json()
      SessionData['Model'] = Model

      # Evaluation Scores
      Scores =dict()
      Scores['train_loss'] = pTrainingSessionResults['scores_all'][i]['train'][0]
      Scores['train_acc'] = pTrainingSessionResults['scores_all'][i]['train'][1]
      Scores['val_loss'] = pTrainingSessionResults['scores_all'][i]['val'][0]
      Scores['val_acc'] = pTrainingSessionResults['scores_all'][i]['val'][1]
      Scores['test_loss'] = pTrainingSessionResults['scores_all'][i]['test'][0]
      Scores['test_acc'] = pTrainingSessionResults['scores_all'][i]['test'][1]
      Scores['train'] = pTrainingSessionResults['scores_all'][i]['train']
      Scores['val'] = pTrainingSessionResults['scores_all'][i]['val']
      Scores['test'] = pTrainingSessionResults['scores_all'][i]['test']            
      SessionData['Scores'] = Scores

      SessionData['TrainingHistory']= pTrainingSessionResults['history_all'][i]

      # Predictions and ROC
      SessionData['Predictions']= pTrainingSessionResults['predictions_all'][i]

      # Session Results Info
      SessionData['SessionResultsInfo']= pTrainingSessionResults['resultsinfo_all'][i]
      
      # Append this training session to the list
      session_data_all.append(SessionData)   
      # end for i Loop for all the folds in the pTrainingSessionResults
      
  return session_data_all


# **** UpdateSessionData_Fold ***
def UpdateSessionData_Fold(pSessionDataAll, pSessionParameters ,pTrainingParameters,pAugParameters, pTrainingSessionResults, pFoldIndex=0):
  '''
  
  - This is a new function to Update only a single fold at a time instead of all folds together to avoid running out of memory 

    *** Input parameters: ***
    pSessionDataAll: List of dict which contains the history of Training Sessions that have been previously recorded to be saved in a json file
    pTrainingSessionResults: All training results and structures from last training. Contains all models,data generators, training scores and history.
   
  '''
  
  VersionNo=1.5
  
  # Check if there is already session data from previous sessions
  if len(pSessionDataAll)==0:
    # Create an empty list for the session data
    session_data_all = []
    FoldGroupSessionNo = 1   # Init Fold Group SessionNo (This was passed manually before but now it will be calculated automatically)
  else:
    # create a reference to the existing session data
    session_data_all = pSessionDataAll
    # Get the Fold Group SessionNo from the previous session and inc by 1
    FoldGroupSessionNo = pSessionDataAll[len(pSessionDataAll)-1]['SessionParameters']['SessionNo'] + 1
    
    
  
  # Create a unique id for this fold group which can be used to group the folds together if needed
  FoldGroupID = str(uuid.uuid4())  
  

  # if this training mode is 'NEW' then create a unique training group id which can be used to group the training sessions of new, resume and finetuning together
  if pSessionParameters['TrainingMode'] =='NEW': 
    # Create a unique id training group id which can be used to group the training sessions of new, resume and finetuning together
    TrainingGroupID = str(uuid.uuid4())      
  else:
    # Get the training group ID from the previous session which was probably NEW or RESUME or RESUME_FINETUNE
    TrainingGroupID = pSessionDataAll[len(pSessionDataAll)-1]['SessionParameters']['TrainingGroupID']
      
  # Only update a single fold  in the pTrainingSessionResults based on the FoldIndex passed as parameter
  i = pFoldIndex
  SessionData = dict()
  SessionData['SessionID'] = str(uuid.uuid4())
  TimeStamp = datetime.datetime.now()
  SessionData['SessionDateTime'] = TimeStamp.strftime("%Y-%m-%d %H:%M")    
  SessionData['TemplateVersionNo'] = VersionNo   

  # Session parameters
  SessionParameters =dict()
  # SessionParameters['SessionNo'] = pSessionParameters['SessionNo']  # This used to be passed manually but now it will be calculated
  SessionParameters['SessionNo'] = FoldGroupSessionNo
  SessionParameters['Description'] = pSessionParameters['Description'] 
  SessionParameters['Comments'] = pSessionParameters['Comments']
  SessionParameters['Evaluation'] = pSessionParameters['Evaluation']
  SessionParameters['TrainingMode'] = pSessionParameters['TrainingMode']    # 1) NEW 2) RESUME 3) FINETUNE  4) RESUME_FINETUNE
  SessionParameters['ModelBuildParameters'] = pSessionParameters['ModelBuildParameters']                
  SessionParameters['BatchNorm'] = pSessionParameters['BatchNorm']    # 1) NONE 2) CONV_ONLY 3) CONV_DENSE (Both Conv and Dense layers)
  SessionParameters['DataAug'] = pSessionParameters['DataAug']    # Flag to indicate whether to use Data Augmentation
  SessionParameters['Finetune_Trainable_Layers'] = pSessionParameters['Finetune_Trainable_Layers']    # Layers set to trainable during finetuning                              
  SessionParameters['TrainTestSplit'] = pSessionParameters['TrainTestSplit']
  SessionParameters['CrossVal'] =  pSessionParameters['CrossVal']
  SessionParameters['NoFolds'] =  pSessionParameters['NoFolds']
  SessionParameters['BestFold'] =  pSessionParameters['BestFold']
  SessionParameters['FoldNo'] =  i+1     # add the current fold no from the loop iteration
  SessionParameters['FoldGroupID'] = FoldGroupID
  SessionParameters['TrainingGroupID'] = TrainingGroupID
  SessionParameters['Verbose'] =  pSessionParameters['Verbose']      
  SessionParameters['DataType'] =  pSessionParameters['DataType']      
  SessionParameters['DataFoldsSource'] =  pSessionParameters['DataFoldsSource']      
  SessionParameters['DataFoldsFileName'] =  pSessionParameters['DataFoldsFileName']                              
  SessionParameters['DataFoldsFileGoogleID'] =  pSessionParameters['DataFoldsFileGoogleID']                                    
  SessionParameters['NoClasses'] =  pSessionParameters['NoClasses']                                          
  SessionParameters['ModelType'] =  pSessionParameters['ModelType']                                          
  SessionParameters['ModelBuildFN'] =  pSessionParameters['ModelBuildFN']                                                
  SessionParameters['ModelFinetuneBuildFN'] =  pSessionParameters['ModelFinetuneBuildFN']                                                            
  SessionData['SessionParameters'] = SessionParameters

  # Training parameters
  SessionData['TrainingParameters']= pTrainingParameters

  # Augmentation parameters
  SessionData['AugParameters']= pAugParameters

  # Model used for training
  Model =dict()
  Model['Configuration'] = dict()
  Model['JSON'] =  pTrainingSessionResults['models_all'][i].to_json()
  SessionData['Model'] = Model

  # Evaluation Scores
  Scores =dict()
  Scores['train_loss'] = pTrainingSessionResults['scores_all'][i]['train'][0]
  Scores['train_acc'] = pTrainingSessionResults['scores_all'][i]['train'][1]
  Scores['val_loss'] = pTrainingSessionResults['scores_all'][i]['val'][0]
  Scores['val_acc'] = pTrainingSessionResults['scores_all'][i]['val'][1]
  Scores['test_loss'] = pTrainingSessionResults['scores_all'][i]['test'][0]
  Scores['test_acc'] = pTrainingSessionResults['scores_all'][i]['test'][1]
  Scores['train'] = pTrainingSessionResults['scores_all'][i]['train']
  Scores['val'] = pTrainingSessionResults['scores_all'][i]['val']
  Scores['test'] = pTrainingSessionResults['scores_all'][i]['test']            
  SessionData['Scores'] = Scores

  # Training History (only record the history from the Keras history object history.history)
  SessionData['TrainingHistory']= pTrainingSessionResults['history_all'][i]

  # Predictions and ROC
  SessionData['Predictions']= pTrainingSessionResults['predictions_all'][i]

  # Session Results Info
  SessionData['SessionResultsInfo']= pTrainingSessionResults['resultsinfo_all'][i]
  
  # Append this training session to the list
  session_data_all.append(SessionData)   
  # end for i Loop for all the folds in the pTrainingSessionResults
      
  return session_data_all




# **** UpdateSessionData_Evaluation ***
def UpdateSessionData_Evaluation(pSessionDataAll, pSessionParameters, pFoldGroupNo):
  '''
  - Updates the evaluation and best fold of the given fold batch for the session data archive without adding new data.
  
    *** Input parameters: ***
    pSessionDataAll: List of dict which contains the history of Training Sessions that have been previously recorded to be saved in a json file
    pSessionParameters: New session parameters with new Comments
    pFoldGroupNo: The fold group no to update       
  '''
    
  NoTrainingSessions = len(pSessionDataAll)
  
  
  FoldBatchCounter =0
  
  i=0
  
  while i < len(pSessionDataAll):  
    FoldBatchCounter += 1  # New Fold Batch


    # Get the number of folds for this first fold batch
    NoFolds = pSessionDataAll[i]['SessionParameters']['NoFolds']       
    FoldCounter = 0
    while FoldCounter < NoFolds:
      FoldCounter += 1  # increment fold counter within this fold batch                 
      # Check to see if this is the target Fold Batch No given as paramter 
      if FoldBatchCounter == pFoldGroupNo:   
        pSessionDataAll[i]['SessionParameters']['Description']  = pSessionParameters['Description'] 
        pSessionDataAll[i]['SessionParameters']['Comments']  = pSessionParameters['Comments'] 
        pSessionDataAll[i]['SessionParameters']['Evaluation']  = pSessionParameters['Evaluation'] 
        pSessionDataAll[i]['SessionParameters']['BestFold']  = pSessionParameters['BestFold'] 
      i +=1  # move to the next training session
      # End while FoldCounter < NoFolds          
   
    # Loop to next fold batch
    # End while i < len(pSessionDataAll)          
  return pSessionDataAll







# **** UpdateSessionData_Comments ***
def UpdateSessionData_Comments(pSessionDataAll, pSessionParameters, pFoldGroupNo):
  '''
  - Updates the Comments of the given fold batch for the session data archive without adding new data.

    *** Input parameters: ***
    pSessionDataAll: List of dict which contains the history of Training Sessions that have been previously recorded to be saved in a json file
    pSessionParameters: New session parameters with new Comments
    pFoldGroupNo: The fold group no to update       
  '''
    
  NoTrainingSessions = len(pSessionDataAll)
  
  
  FoldBatchCounter =0
  
  i=0
  
  while i < len(pSessionDataAll):  
    FoldBatchCounter += 1  # New Fold Batch


    # Get the number of folds for this first fold batch
    NoFolds = pSessionDataAll[i]['SessionParameters']['NoFolds']       
    FoldCounter = 0
    while FoldCounter < NoFolds:
      FoldCounter += 1  # increment fold counter within this fold batch                 
      # Check to see if this is the target Fold Batch No given as paramter 
      if FoldBatchCounter == pFoldGroupNo:   
        pSessionDataAll[i]['SessionParameters']['Comments']  = pSessionParameters['Comments'] 
      i +=1  # move to the next training session
      # End while FoldCounter < NoFolds          
   
    # Loop to next fold batch
    # End while i < len(pSessionDataAll)          
  return pSessionDataAll



# **** UpdateSessionData_Comments ***
def UpdateSessionData_ReNumberFoldGroupSessionNo(pSessionDataAll):
  '''
  - This FN is for merging several lists for SessionDataArchiveAll that resulted from different Pipeline batch runs. 

    *** Input parameters: ***
    pSessionDataAll: List of dict which contains the history of Training Sessions that have been previously recorded to be saved in a json file
  '''
    
  
  FoldBatchCounter =0
  
  i=0
  
  while i < len(pSessionDataAll):  
    FoldBatchCounter += 1  # New Fold Batch


    # Get the number of folds for this first fold batch
    if 'NoFoldIterations' in pSessionDataAll[i]['TrainingParameters'].keys():
        NoFolds = pSessionDataAll[i]['TrainingParameters']['NoFoldIterations'] 
    else:
        NoFolds = pSessionDataAll[i]['SessionParameters']['NoFolds']
    
    FoldCounter = 0
    while FoldCounter < NoFolds:
      FoldCounter += 1  # increment fold counter within this fold batch                 
      # Check to see if this Fold Batch No is not correct
      if pSessionDataAll[i]['SessionParameters']['SessionNo']  != FoldBatchCounter:             
        print('Corrected index '+str(i)+ '  From: ' + str(pSessionDataAll[i]['SessionParameters']['SessionNo']) + '  To: '+str(FoldBatchCounter) )
        # Correct FoldGroupSessionNo
        pSessionDataAll[i]['SessionParameters']['SessionNo']  = FoldBatchCounter 
      i +=1  # move to the next training session
      # End while FoldCounter < NoFolds          
   
    # Loop to next fold batch
    # End while i < len(pSessionDataAll)          
  return pSessionDataAll



def SaveSessionData(pSessionDataAll):  
  '''
  - Saves the current session data into a new json file and returns the filename
  - The filename is generated automatically based on current date and time
  '''

  for i in range(len(pSessionDataAll)):
      # Check if LR is part of the history
      if 'lr' in pSessionDataAll[i]['TrainingHistory'].keys():
          # Convert LR to float64 instead of float32
          pSessionDataAll[i]['TrainingHistory']['lr'] = [np.float64(lr) for lr in pSessionDataAll[i]['TrainingHistory']['lr']] 
         
      # Convert acc to float64 instead of float32
      pSessionDataAll[i]['TrainingHistory']['acc'] = [np.float64(acc) for acc in pSessionDataAll[i]['TrainingHistory']['acc']] 

  '''
  *** Convert ndim numpy arrays of type numpy.ndarray to lists so they can be be serialized directly by json.dump ***
  - The predictions data produced for the ROC curve analysis is of type numpy.ndarray which cannot be serialized directly by json.dump.
  - The solution is to convert these items into lists before saving the json file and then convert them back to numpy arrays when loading the file so
  '''
      
      # Check if predictions have been calculated
      if len(pSessionDataAll[i]['Predictions'])!=0:
          pSessionDataAll[i]['Predictions']['train']['y_predictions'] = pSessionDataAll[i]['Predictions']['train']['y_predictions'].tolist()          
          pSessionDataAll[i]['Predictions']['train']['class_predictions'] = pSessionDataAll[i]['Predictions']['train']['class_predictions'].tolist()          
          pSessionDataAll[i]['Predictions']['train']['ROC_FPR'] = pSessionDataAll[i]['Predictions']['train']['ROC_FPR'].tolist()
          pSessionDataAll[i]['Predictions']['train']['ROC_TPR'] = pSessionDataAll[i]['Predictions']['train']['ROC_TPR'].tolist()
          pSessionDataAll[i]['Predictions']['train']['ROC_thresholds'] = pSessionDataAll[i]['Predictions']['train']['ROC_thresholds'].tolist()
          pSessionDataAll[i]['Predictions']['val']['y_predictions'] = pSessionDataAll[i]['Predictions']['val']['y_predictions'].tolist()          
          pSessionDataAll[i]['Predictions']['val']['class_predictions'] = pSessionDataAll[i]['Predictions']['val']['class_predictions'].tolist()          
          pSessionDataAll[i]['Predictions']['val']['ROC_FPR'] = pSessionDataAll[i]['Predictions']['val']['ROC_FPR'].tolist()
          pSessionDataAll[i]['Predictions']['val']['ROC_TPR'] = pSessionDataAll[i]['Predictions']['val']['ROC_TPR'].tolist()
          pSessionDataAll[i]['Predictions']['val']['ROC_thresholds'] = pSessionDataAll[i]['Predictions']['val']['ROC_thresholds'].tolist()
          pSessionDataAll[i]['Predictions']['test']['y_predictions'] = pSessionDataAll[i]['Predictions']['test']['y_predictions'].tolist()          
          pSessionDataAll[i]['Predictions']['test']['class_predictions'] = pSessionDataAll[i]['Predictions']['test']['class_predictions'].tolist()          
          pSessionDataAll[i]['Predictions']['test']['ROC_FPR'] = pSessionDataAll[i]['Predictions']['test']['ROC_FPR'].tolist()
          pSessionDataAll[i]['Predictions']['test']['ROC_TPR'] = pSessionDataAll[i]['Predictions']['test']['ROC_TPR'].tolist()
          pSessionDataAll[i]['Predictions']['test']['ROC_thresholds'] = pSessionDataAll[i]['Predictions']['test']['ROC_thresholds'].tolist()          
  

  DataType = pSessionDataAll[len(pSessionDataAll)-1]['SessionParameters']['DataType']
  ModelType = pSessionDataAll[len(pSessionDataAll)-1]['SessionParameters']['ModelType']
  ModelType = pSessionDataAll[len(pSessionDataAll)-1]['SessionParameters']['ModelBuildParameters']['Model']


  now = datetime.datetime.now()
  if gPipelineModeFlag:  
      FileName = gPipelineBatchNo + '_'+ DataType+ '_' + ModelType+ '_TrainHistory_('+str(now.year)+'_'+str(now.month)+'_'+str(now.day)+')_('+str(now.hour)+'_'+str(now.minute)+'_'+str(now.second)+').json'    
  else:
      FileName = DataType+ '_' + ModelType+ '_TrainHistory_('+str(now.year)+'_'+str(now.month)+'_'+str(now.day)+')_('+str(now.hour)+'_'+str(now.minute)+'_'+str(now.second)+').json'  
      

  with open(FileName, "w") as write_file:
      json.dump(pSessionDataAll, write_file,indent=0)
      
  '''    
  *** Convert lists back into ndim numpy arrays of type numpy.ndarray so they can be processed normaly ***
  '''
  for i in range(len(pSessionDataAll)):
      if len(pSessionDataAll[i]['Predictions'])!=0:
          pSessionDataAll[i]['Predictions']['train']['y_predictions'] = np.array(pSessionDataAll[i]['Predictions']['train']['y_predictions'])
          pSessionDataAll[i]['Predictions']['train']['class_predictions'] = np.array(pSessionDataAll[i]['Predictions']['train']['class_predictions'])
          pSessionDataAll[i]['Predictions']['train']['ROC_FPR'] = np.array(pSessionDataAll[i]['Predictions']['train']['ROC_FPR'])
          pSessionDataAll[i]['Predictions']['train']['ROC_TPR'] = np.array(pSessionDataAll[i]['Predictions']['train']['ROC_TPR'])
          pSessionDataAll[i]['Predictions']['train']['ROC_thresholds'] = np.array(pSessionDataAll[i]['Predictions']['train']['ROC_thresholds'])
          pSessionDataAll[i]['Predictions']['val']['y_predictions'] = np.array(pSessionDataAll[i]['Predictions']['val']['y_predictions'])
          pSessionDataAll[i]['Predictions']['val']['class_predictions'] = np.array(pSessionDataAll[i]['Predictions']['val']['class_predictions'])
          pSessionDataAll[i]['Predictions']['val']['ROC_FPR'] = np.array(pSessionDataAll[i]['Predictions']['val']['ROC_FPR'])
          pSessionDataAll[i]['Predictions']['val']['ROC_TPR'] = np.array(pSessionDataAll[i]['Predictions']['val']['ROC_TPR'])
          pSessionDataAll[i]['Predictions']['val']['ROC_thresholds'] = np.array(pSessionDataAll[i]['Predictions']['val']['ROC_thresholds'])
          pSessionDataAll[i]['Predictions']['test']['y_predictions'] = np.array(pSessionDataAll[i]['Predictions']['test']['y_predictions'])
          pSessionDataAll[i]['Predictions']['test']['class_predictions'] = np.array(pSessionDataAll[i]['Predictions']['test']['class_predictions'])
          pSessionDataAll[i]['Predictions']['test']['ROC_FPR'] = np.array(pSessionDataAll[i]['Predictions']['test']['ROC_FPR'])
          pSessionDataAll[i]['Predictions']['test']['ROC_TPR'] = np.array(pSessionDataAll[i]['Predictions']['test']['ROC_TPR'])
          pSessionDataAll[i]['Predictions']['test']['ROC_thresholds'] = np.array(pSessionDataAll[i]['Predictions']['test']['ROC_thresholds'])          

  FileGoogleID = ''
  if gUploadFilesToGoogleDriveFlag:
      FileGoogleID = UploadFileToGoogleDrive(FileName, gGoogleDrive_SessionDataFolderID) 

  print('Successfuly saved Session Data to: '+FileName)
  
  return FileName, FileGoogleID


# **** LoadSessionData ***
def LoadSessionData(pFileName ,pFileGoogleID):
  '''
  - Loads all the training session data into a list of dictionaries from the saved json file. 
  '''
  
  # Download the file from google drive if Google Drive Flag is set to True (Default)
  if gLoadFilesFromGoogleDriveFlag:      
    DownloadFileFromGoogleDrive(pFileName, pFileGoogleID)  
  
  with open(pFileName, "r") as read_file:
    SessionDataArchiveAll = json.load(read_file)  
    
  '''
  *** Convert lists into ndim numpy arrays of type numpy.ndarray so they can be processed normaly ***
  - The predictions data produced for the ROC curve analysis is of type numpy.ndarray which cannot be serialized directly by json.dump.
  - The solution is to convert these items into lists before saving the json file and then convert them back to numpy arrays when loading the file so
  '''
  for i in range(len(SessionDataArchiveAll)):
      if 'Predictions' in SessionDataArchiveAll[i].keys():   
          if len(SessionDataArchiveAll[i]['Predictions'])!=0:
              SessionDataArchiveAll[i]['Predictions']['train']['y_predictions'] = np.array(SessionDataArchiveAll[i]['Predictions']['train']['y_predictions'])
              SessionDataArchiveAll[i]['Predictions']['train']['class_predictions'] = np.array(SessionDataArchiveAll[i]['Predictions']['train']['class_predictions'])
              SessionDataArchiveAll[i]['Predictions']['train']['ROC_FPR'] = np.array(SessionDataArchiveAll[i]['Predictions']['train']['ROC_FPR'])
              SessionDataArchiveAll[i]['Predictions']['train']['ROC_TPR'] = np.array(SessionDataArchiveAll[i]['Predictions']['train']['ROC_TPR'])
              SessionDataArchiveAll[i]['Predictions']['train']['ROC_thresholds'] = np.array(SessionDataArchiveAll[i]['Predictions']['train']['ROC_thresholds'])
              SessionDataArchiveAll[i]['Predictions']['val']['y_predictions'] = np.array(SessionDataArchiveAll[i]['Predictions']['val']['y_predictions'])
              SessionDataArchiveAll[i]['Predictions']['val']['class_predictions'] = np.array(SessionDataArchiveAll[i]['Predictions']['val']['class_predictions'])
              SessionDataArchiveAll[i]['Predictions']['val']['ROC_FPR'] = np.array(SessionDataArchiveAll[i]['Predictions']['val']['ROC_FPR'])
              SessionDataArchiveAll[i]['Predictions']['val']['ROC_TPR'] = np.array(SessionDataArchiveAll[i]['Predictions']['val']['ROC_TPR'])
              SessionDataArchiveAll[i]['Predictions']['val']['ROC_thresholds'] = np.array(SessionDataArchiveAll[i]['Predictions']['val']['ROC_thresholds'])
              SessionDataArchiveAll[i]['Predictions']['test']['y_predictions'] = np.array(SessionDataArchiveAll[i]['Predictions']['test']['y_predictions'])
              SessionDataArchiveAll[i]['Predictions']['test']['class_predictions'] = np.array(SessionDataArchiveAll[i]['Predictions']['test']['class_predictions'])
              SessionDataArchiveAll[i]['Predictions']['test']['ROC_FPR'] = np.array(SessionDataArchiveAll[i]['Predictions']['test']['ROC_FPR'])
              SessionDataArchiveAll[i]['Predictions']['test']['ROC_TPR'] = np.array(SessionDataArchiveAll[i]['Predictions']['test']['ROC_TPR'])
              SessionDataArchiveAll[i]['Predictions']['test']['ROC_thresholds'] = np.array(SessionDataArchiveAll[i]['Predictions']['test']['ROC_thresholds'])    
    
  print("Successfully loaded total: "+str(len(SessionDataArchiveAll))+" Sessions")  
        
  return SessionDataArchiveAll



# **** LoadSessionData ***
def LoadSessionData_Merged(pFileName ,pFileGoogleID):
  '''
  - Loads and Merges all the training session data into a list of dictionaries from the saved json file. 
  - This FN is used to Load and merge several lists for SessionDataArchiveAll that resulted from different Pipeline batch runs. 
  
    *** Input parameters: ***
    pFileName: List of filenames which contains the history of Training Sessions that have been previously recorded in a json file
    pFileGoogleID: List of GoogleDrive ID coresponding to the list of files for Training Sessions 
  '''    

  # Init Merged SessionData     
  SessionDataArchiveAll = []

  # Loop for all the SessionDataArchive files passed as a parameter to load and Merge them
  for k in range(len(pFileName)):
      SessionDataArchive = LoadSessionData(pFileName[k], pFileGoogleID[k])
      # Merge SessionArchive
      SessionDataArchiveAll = SessionDataArchiveAll + SessionDataArchive
      
      # End for k in range(len(pFileName)):
  
  # Renumber the FoldGroupSessionNo from start to end so that the values are correct and consecutive.
  UpdateSessionData_ReNumberFoldGroupSessionNo(SessionDataArchiveAll)  
        
  return SessionDataArchiveAll


# **** SaveSessionModels ***
def SaveSessionModels(pTrainingSessionResults, pSessionDataAll, pFilterFoldsFlag = False, pFilterFolds=[]):
  '''
  - Saves the full session Models and uploads the files in GoogleDrive
  - Saving the full model includes: 
      1) Model architecture allowingto re-create the model
      2) Model weights
      3) Training configuration (loss, optimizer)
      4) State of the optimizer, allowing to resume training exactly where you left off  

    *** Input parameters: ***
    pTrainingSessionResults: All training results and structures from last training. Contains all models,data generators, training scores and history.
    pSessionDataAll: List of dict which contains the history of all Training Sessions that have been previously recorded including the current one to be saved
    pFilterFoldsFlag: Whether or not to filter the folds to be saved 
    pFilterFolds: List with folds to be saved starting at 1 if pFilterFoldsFlag is TRUE. For example [1,2,3]
  '''
 
  # Get the IDs for the current session 
  TrainingGroupID = pSessionDataAll[len(pSessionDataAll)-1]['SessionParameters']['TrainingGroupID']
  DataType = pSessionDataAll[len(pSessionDataAll)-1]['SessionParameters']['DataType'] 
  ModelType = pSessionDataAll[len(pSessionDataAll)-1]['SessionParameters']['ModelBuildParameters']['Model']  
 
 
  if 'NoFoldIterations' in pSessionDataAll[len(pSessionDataAll)-1]['TrainingParameters'].keys():
      NoFolds = pSessionDataAll[len(pSessionDataAll)-1]['TrainingParameters']['NoFoldIterations'] 
  else:
      NoFolds = pSessionDataAll[len(pSessionDataAll)-1]['SessionParameters']['NoFolds']   
    
  
  FoldGroupID = pSessionDataAll[len(pSessionDataAll)-1]['SessionParameters']['FoldGroupID']
  FoldGroupSessionNo = pSessionDataAll[len(pSessionDataAll)-1]['SessionParameters']['SessionNo']       
  SessionID = pSessionDataAll[len(pSessionDataAll)-1]['SessionID'] 
  SessionDateTime = pSessionDataAll[len(pSessionDataAll)-1]['SessionDateTime'] 
  
  FilteredSessionDataArchiveAll = [querylist for querylist in pSessionDataAll if querylist['SessionParameters']['TrainingGroupID']==TrainingGroupID ]

  # Update the TrainingGroup Flag for all the filtered list
  for i in range(len(FilteredSessionDataArchiveAll)):
    FilteredSessionDataArchiveAll[i]['SessionResultsInfo']['TrainingGroupModelFlag'] = True

   
  # Loop for all the folds in the pTrainingSessionResults
  for i in range(len(pTrainingSessionResults['models_all'])):
    # Check if this Fold should be Saved from the filter parameters 
    if (not pFilterFoldsFlag) or (pFilterFoldsFlag and i+1 in pFilterFolds):   

      # Get the current Fold Session Index in the to pSessionDataAll List 
      FoldIndex = len(pSessionDataAll) - NoFolds + i   # Fold index with base 0
      FoldSessionCounter = FoldIndex + 1   # Fold Session Counter with base 1
      
      pTrainingSessionResults['resultsinfo_all'][i]['TrainingGroupModelFlag'] = True
      
      pTrainingSessionResults['resultsinfo_all'][i]['SessionModelFlag'] = True
      pSessionDataAll[FoldIndex]['SessionResultsInfo']['SessionModelFlag'] = True
      
      # Get loss and acc
      loss = pTrainingSessionResults['scores_all'][i]['val'][0]
      acc = pTrainingSessionResults['scores_all'][i]['val'][1]    


      now = datetime.datetime.now()
      DateStr= '('+str(now.year)+'_'+str(now.month)+'_'+str(now.day)+')_('+str(now.hour)+'_'+str(now.minute)+'_'+str(now.second)+')'
      FileIndexStr = '{:0>6d}'.format(FoldSessionCounter)  # Left padding with total 6 digits
      LossAccStr = 'Loss {:.5f} Acc {:.2f}'.format(loss,acc)
      if gPipelineModeFlag:
          FileNameModel = gPipelineBatchNo +'_'+ DataType + '_' + ModelType+'_Model_'+FileIndexStr+'_Fold'+str(i+1)+'_'+LossAccStr+'_'+DateStr+'.h5'        
      else:    
          FileNameModel = 'GE '+DataType+'_Model_'+FileIndexStr+'_Fold'+str(i+1)+' '+LossAccStr+'_'+DateStr+'.h5'            
      
      # Save Model File
      pTrainingSessionResults['models_all'][i].save(FileNameModel)    

      # Upload to GoogleDrive
      if gUploadFilesToGoogleDriveFlag:
          GoogleIDModel = UploadFileToGoogleDrive(FileNameModel, gGoogleDrive_ModelDataFolderID) 
      else:
          GoogleIDModel = ''
 
      # Set the Model FileName and GoogleID for this fold session in both the TrainingSessionResults and the SessionDataAll
      pTrainingSessionResults['resultsinfo_all'][i]['ModelFileName'] = FileNameModel
      pTrainingSessionResults['resultsinfo_all'][i]['ModelFileGoogleID'] = GoogleIDModel
      pSessionDataAll[FoldIndex]['SessionResultsInfo']['ModelFileName'] = FileNameModel
      pSessionDataAll[FoldIndex]['SessionResultsInfo']['ModelFileGoogleID'] = GoogleIDModel
      

      print('Successfuly saved Model for Fold '+str(i+1) +':  '+FileNameModel)    

      # end if fold is filtered

    # end for i Loop for all the folds in the pTrainingSessionResults
      
  return



# **** SaveSessionWeights ***
def SaveSessionWeights(pTrainingSessionResults, pSessionDataAll, pFilterFoldsFlag = False, pFilterFolds=[]):
  '''
  - Saves only the Model Weights and uploads the files in GoogleDrive

    *** Input parameters: ***
    pTrainingSessionResults: All training results and structures from last training. Contains all models,data generators, training scores and history.
    pSessionDataAll: List of dict which contains the history of all Training Sessions that have been previously recorded including the current one to be saved
    pFilterFoldsFlag: Whether or not to filter the folds to be saved 
    pFilterFolds: List with folds to be saved starting at 1 if pFilterFoldsFlag is TRUE. For example [1,2,3]
  '''
 
  # Get the IDs for the current session 
  TrainingGroupID = pSessionDataAll[len(pSessionDataAll)-1]['SessionParameters']['TrainingGroupID']
  DataType = pSessionDataAll[len(pSessionDataAll)-1]['SessionParameters']['DataType']
  ModelType = pSessionDataAll[len(pSessionDataAll)-1]['SessionParameters']['ModelBuildParameters']['Model']
  
  
  if 'NoFoldIterations' in pSessionDataAll[len(pSessionDataAll)-1]['TrainingParameters'].keys():
      NoFolds = pSessionDataAll[len(pSessionDataAll)-1]['TrainingParameters']['NoFoldIterations'] 
  else:
      NoFolds = pSessionDataAll[len(pSessionDataAll)-1]['SessionParameters']['NoFolds']   
  
  
  FoldGroupID = pSessionDataAll[len(pSessionDataAll)-1]['SessionParameters']['FoldGroupID']
  FoldGroupSessionNo = pSessionDataAll[len(pSessionDataAll)-1]['SessionParameters']['SessionNo']       
  SessionID = pSessionDataAll[len(pSessionDataAll)-1]['SessionID'] 
  SessionDateTime = pSessionDataAll[len(pSessionDataAll)-1]['SessionDateTime'] 
  

  FilteredSessionDataArchiveAll = [querylist for querylist in pSessionDataAll if querylist['SessionParameters']['TrainingGroupID']==TrainingGroupID ]

  # Update the TrainingGroup Flag for all the filtered list
  for i in range(len(FilteredSessionDataArchiveAll)):
    FilteredSessionDataArchiveAll[i]['SessionResultsInfo']['TrainingGroupWeightsFlag'] = True

  
  # Init the value of best session ID and FoldIndex in all the training group  
  TrainingGroupBestSessionID=''
  TrainingGroupBestSessionFoldIndex=0
    
  # Loop for all the folds in the pTrainingSessionResults
  for i in range(len(pTrainingSessionResults['models_all'])):
    # Check if this Fold should be Saved from the filter parameters 
    if (not pFilterFoldsFlag) or (pFilterFoldsFlag and i+1 in pFilterFolds):   

      # Check if saving weights of intermediate results  
      if len(pSessionDataAll) < NoFolds:
          FoldIndex = i  
      else:

          FoldIndex = len(pSessionDataAll) - NoFolds + i   # Fold index with base 0
     
      FoldSessionCounter = FoldIndex + 1   # Fold Session Counter with base 1
      

      pTrainingSessionResults['resultsinfo_all'][i]['TrainingGroupWeightsFlag'] = True              
      
      pTrainingSessionResults['resultsinfo_all'][i]['SessionWeightsFlag'] = True
      pSessionDataAll[FoldIndex]['SessionResultsInfo']['SessionWeightsFlag'] = True
      
      # Get loss and acc
      loss = pTrainingSessionResults['scores_all'][i]['val'][0]
      acc = pTrainingSessionResults['scores_all'][i]['val'][1]    


      now = datetime.datetime.now()
      DateStr= '('+str(now.year)+'_'+str(now.month)+'_'+str(now.day)+')_('+str(now.hour)+'_'+str(now.minute)+'_'+str(now.second)+')'
      FileIndexStr = '{:0>6d}'.format(FoldSessionCounter)  # Left padding with total 6 digits
      LossAccStr = 'Loss {:.5f} Acc {:.2f}'.format(loss,acc)
      if gPipelineModeFlag:
          FileNameWeights = gPipelineBatchNo +'_'+ DataType + '_' + ModelType+'_Weights_'+FileIndexStr+'_Fold'+str(i+1)+'_'+LossAccStr+'_'+DateStr+'.h5'        
      else:    
          FileNameWeights = 'GE '+DataType+'_Weights_'+FileIndexStr+'_Fold'+str(i+1)+' '+LossAccStr+'_'+DateStr+'.h5'            
      
      # Save Weight Files
      pTrainingSessionResults['models_all'][i].save_weights(FileNameWeights)    

      # Upload to GoogleDrive
      if gUploadFilesToGoogleDriveFlag:
          GoogleIDWeights = UploadFileToGoogleDrive(FileNameWeights, gGoogleDrive_WeightDataFolderID) 
      else:
          GoogleIDWeights = ''
            
      
      # Set the Weight FileName and GoogleID for this fold session in both the TrainingSessionResults and the SessionDataAll
      pTrainingSessionResults['resultsinfo_all'][i]['WeightsFileName'] = FileNameWeights
      pTrainingSessionResults['resultsinfo_all'][i]['WeightsFileGoogleID'] = GoogleIDWeights            
      pSessionDataAll[FoldIndex]['SessionResultsInfo']['WeightsFileName'] = FileNameWeights
      pSessionDataAll[FoldIndex]['SessionResultsInfo']['WeightsFileGoogleID'] = GoogleIDWeights

      if pSessionDataAll[FoldIndex]['SessionParameters']['BestFold'] == i+1:
        # Get the SessionID 
        TrainingGroupBestSessionID = pSessionDataAll[FoldIndex]['SessionID']
        # Get the FoldIndex 
        TrainingGroupBestSessionFoldIndex = FoldIndex

      print('Successfuly saved Weights for Fold '+str(i+1) +':  '+FileNameWeights)    
      # end if fold is filtered

    # end for i Loop for all the folds in the pTrainingSessionResults

  FilteredSessionDataArchiveAll = [querylist for querylist in pSessionDataAll if querylist['SessionParameters']['TrainingGroupID']==TrainingGroupID ]

  # Update the TrainingGroupBestSessionID and TrainingGroupBestSessionFlag for all the filtered list
  for i in range(len(FilteredSessionDataArchiveAll)):
    FilteredSessionDataArchiveAll[i]['SessionResultsInfo']['TrainingGroupBestSessionID'] = TrainingGroupBestSessionID
    FilteredSessionDataArchiveAll[i]['SessionResultsInfo']['TrainingGroupBestSessionFlag'] = ''  # Reset all Sessions to not be the best in this Training Group

  # Update the TrainingGroupBestSessionFlag for only the best session in the Training Group
  pSessionDataAll[TrainingGroupBestSessionFoldIndex]['SessionResultsInfo']['TrainingGroupBestSessionFlag'] = True  # Set best Session in this Training Group  
      
  return




def SaveData_TrainTestSplit( px_train_aug, px_test_aug, py_train_aug, py_test_aug, py_train_aug_multiclass, py_test_aug_multiclass, pFileNamePrefix='GE Data_', pSplit=0.15, pFilterList=[], pMultiClassLabelsFlag = False):  
  '''
  Function to save the data which has been shuffled and split into Train and Test

  Input parameters:
    px_train_aug: training data 
    px_test_aug: test data     
    py_train_aug:  training labels
    py_test_aug:  test labels    
    pFileNamePrefix: A prefix for the produced files. Example: 'TCGA Breast Data_'
    pSplit: Split for percentage for training and test data
    pFilterList: Any filter list used to filter the data before saving so that it is recorded as info and we dont loose it
    
  Return paramters:
  '''     
    
  # Get the training and test data     
  x_train = px_train_aug  
  x_test = px_test_aug  
  y_train = py_train_aug  
  y_test = py_test_aug 
  y_train_multiclass = py_train_aug_multiclass  
  y_test_multiclass = py_test_aug_multiclass 
   
   
  # *** Training Data ***
  df_x_train = pd.DataFrame(x_train)  # Read train data        
  df_x_test = pd.DataFrame(x_test)   # Read test data
  # Append test data to main data
  df_x_train = df_x_train.append(df_x_test) 
  
  # *** Label Data ***
  df_y_train = pd.DataFrame(y_train)  # Read train data        
  df_y_test = pd.DataFrame(y_test)   # Read test data
  # Append test data to main data
  df_y_train = df_y_train.append(df_y_test)      

  # *** Multiclass Label Data ***
  if pMultiClassLabelsFlag:
      df_y_train_multiclass = pd.DataFrame(y_train_multiclass)  # Read train data        
      df_y_test_multiclass = pd.DataFrame(y_test_multiclass)   # Read test data
      # Append test data to main data
      df_y_train_multiclass = df_y_train_multiclass.append(df_y_test_multiclass)    

    
    
  # *** Save DataFolds Info in a seperate json file so the data can be loaded correctly ***
  DataInfo = dict()   
  DataInfo['TrainTestSplit'] = pSplit
  DataInfo['train_len'] = x_train.shape[0] 
  DataInfo['test_len'] = x_test.shape[0]
  DataInfo['FilterList'] = pFilterList
  DataInfo['MultiClassFlag'] = pMultiClassLabelsFlag

  # *** SAVE All Files and Upload to Google Drive ***
  import datetime
  import json

  print('Saving Data Files...')    

  now = datetime.datetime.now()
  FileNameSuffix = '('+str(now.year)+'_'+str(now.month)+'_'+str(now.day)+')_('+str(now.hour)+'_'+str(now.minute)+'_'+str(now.second)+')'

  print('Saving Training Data...')             
  FileName_x = pFileNamePrefix + 'ShuffledSplit_TRAINING_COMPRESSED_'+ FileNameSuffix + '.txt' 
  df_x_train.to_csv(FileName_x, header=False, index=False, compression='zip')
  print('Successfuly saved Shuffled Split TRAINING Data to: '+FileName_x)  
    
  print('Saving Binary Label Data...')       
  # Default label for Binary Classs
  FileName_y = pFileNamePrefix + 'ShuffledSplit_LABELS_'+ FileNameSuffix + '.txt'
  df_y_train.to_csv(FileName_y, header=False, index=False)
  print('Successfuly saved Shuffled Split Binary LABELS to: '+FileName_y)    

  # Save Multiclass Labels if given Flag = True
  if pMultiClassLabelsFlag:      
      # update the Label for Multi class
      FileName_y_multiclass = pFileNamePrefix + 'ShuffledSplit_LABELS_MULTICLASS_'+ FileNameSuffix + '.txt'
      df_y_train_multiclass.to_csv(FileName_y_multiclass, header=False, index=False)
      print('Successfuly saved Shuffled Split MultiClass LABELS to: '+FileName_y_multiclass)  
  else:       
      FileName_y_multiclass=''

  print('Saving Info Data...')    
  FileName_info = pFileNamePrefix + 'ShuffledSplit_INFO_'+ FileNameSuffix + '.json' 
  
  with open(FileName_info, "w") as write_file:
      json.dump(DataInfo, write_file)
      
  print('Successfuly saved Shuffled Split Info to: '+FileName_info)  

  FileName_x_GoogleID, FileName_y_GoogleID, FileName_y_multiclass_GoogleID, FileName_info_GoogleID ='','','',''
  
  if gUploadFilesToGoogleDriveFlag:
      print('Uploading files to Google Drive...')          
      FileName_x_GoogleID = UploadFileToGoogleDrive(FileName_x, gGoogleDrive_DataFoldsFolderID )
      FileName_y_GoogleID = UploadFileToGoogleDrive(FileName_y, gGoogleDrive_DataFoldsFolderID )            
      FileName_info_GoogleID = UploadFileToGoogleDrive(FileName_info, gGoogleDrive_DataFoldsFolderID )      
      if pMultiClassLabelsFlag:      
          FileName_y_multiclass_GoogleID = UploadFileToGoogleDrive(FileName_y_multiclass, gGoogleDrive_DataFoldsFolderID )            
          
      print('Successfuly uploaded files to Google Drive')        
  
  return FileName_x, FileName_y,FileName_y_multiclass, FileName_info, FileName_x_GoogleID, FileName_y_GoogleID,FileName_y_multiclass_GoogleID, FileName_info_GoogleID




def LoadData_TrainTestSplit( pFileName_x, pFileName_y, pFileName_info):
  ''' 
  - This function loads the data which has already been shuffled and split into training and test based on psplit percentage. 
 
  Input parameters:
    pFileName_x: training and test data 
    pFileName_y: training and test labels
    pFileName_info: info file 
   
  Return paramters:
    x_train_aug, x_test_aug, y_train_aug, y_test_aug, TrainTestSplit  : Training and Test data and TrainTestSplit 
  '''     

  # Load data and Labels  
  print("Loading TrainTestSplit Data...")   
  # Check if the data file is compressed by having the word "COMPRESSED" in the FileName
  if 'COMPRESSED' in pFileName_x:
      print("Loading Compressed Data File...")   
      df_x_train = pd.read_csv(pFileName_x , header=None, compression='zip')
  else:
      df_x_train = pd.read_csv(pFileName_x , header=None)
        
  df_y_train = pd.read_csv(pFileName_y , header=None)

  # Load info file
  with open(pFileName_info, "r") as read_file:
    DataInfo = json.load(read_file)     
  
  # Convert daya to numpy arrays
  x_train_ALL = df_x_train.values[:,:]
  y_train_ALL = df_y_train.values[:,:]

  # Split the Data into Training and Test as it was before saving
  x_train_norm_shuffled = x_train_ALL[ : DataInfo['train_len'] , : ]   # Rows from 0 to Train_len (excluded) and all columns
  x_test_norm_shuffled = x_train_ALL[ DataInfo['train_len'] : , : ]   # Rows from Train_len till end and all columns

  y_train_shuffled = y_train_ALL[ : DataInfo['train_len'] , : ]
  y_test_shuffled = y_train_ALL[ DataInfo['train_len'] : , : ]  

  print("Successfully loaded Data")    

  # Display shape and Class label percentage information for original data
  print ('x_original:' + str(x_train_ALL.shape))
  print ('y_original:' + str(y_train_ALL.shape)) 

  print('')
  print('Train/Test Split: '+str(DataInfo['TrainTestSplit']))
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


  return x_train_aug, x_test_aug, y_train_aug, y_test_aug , DataInfo['TrainTestSplit']
 





def GetClassPredictionPercentages( pSessionDataArchiveAll, pSessionDataIndex, pFoldNo, pLabelsFileName, pLabelsGoogleID, pNClasses):
  ''' 

  
  Input parameters:
    pSessionDataArchiveAll: Session Archive 
    pSessionDataIndex: The SessionIndex to use from the pSessionDataArchiveAll to get the Pred percentages
    pFoldNo: Fold No for fold to be calculated (Ex: 1, 2, 3, 4, 5)
    pLabeslFileName: Multiclass labels Filename 
    pLabelsGoogleID: Multiclass labels GoogleDriveID
    pNClasses: No of Classes
   
  Return paramters:
    ClassPredictionPercentages : Array with Accuracy for each Class
  '''     

  # Download Label File
  if gLoadFilesFromGoogleDriveFlag:      
    DownloadFileFromGoogleDrive(pLabelsFileName, pLabelsGoogleID) 
       
  dfLabels = pd.read_csv(pLabelsFileName, header=None)
  print('Labels Shape: '+str(dfLabels.shape))
  predictions_train = pSessionDataArchiveAll[pSessionDataIndex]['Predictions']['train']['class_predictions']
  predictions_val = pSessionDataArchiveAll[pSessionDataIndex]['Predictions']['val']['class_predictions']
  predictions_test = pSessionDataArchiveAll[pSessionDataIndex]['Predictions']['test']['class_predictions']

  print('Shapes')
  print(predictions_train.shape)
  print(predictions_val.shape)
  print(predictions_test.shape)
  print('Original Scores:')
  print(pSessionDataArchiveAll[pSessionDataIndex]['Scores']['train_acc'])
  print(pSessionDataArchiveAll[pSessionDataIndex]['Scores']['val_acc'])
  print(pSessionDataArchiveAll[pSessionDataIndex]['Scores']['test_acc']) 
    
  
  # Convert data to numpy arrays
  y_ALL = dfLabels.values[:,:]
  y_train_val = y_ALL[ : predictions_train.shape[0]+predictions_val.shape[0] , : ]    
  y_test  = y_ALL[ predictions_train.shape[0]+predictions_val.shape[0] : , : ]
    
  n_val = predictions_val.shape[0]
    
  # Set the Fold to be created based on the given index passed as parameter
  i = pFoldNo-1
  print('******************************************'+'\n')
  print('Creating fold # ', i+1)
  # Prepare the validation data: data from partition # k
  print ('y_val ' + str(i * n_val)+':'+ str((i + 1) * n_val) )
  y_val = y_train_val[i * n_val: (i + 1) * n_val ]
  # Prepare the training data: data from all other partitions
  print ('y_train Concat [:' + str(i * n_val)+'] + ['+ str((i + 1) * n_val)+':]' )
  y_train = np.concatenate([y_train_val[:i * n_val],y_train_val[(i + 1) * n_val:]], axis=0)


  print("Successfully loaded Label Data")    

  # Display shape and Class label percentage information for original data
  print ('y_train:' + str(y_train.shape))
  print ('y_val:' + str(y_val.shape))
  print ('y_test:' + str(y_test.shape))
  
  ClassCorrectCount_train = np.zeros([pNClasses])
  ClassAcc_train = np.zeros([pNClasses])
  ClassCorrectCount_val = np.zeros([pNClasses])
  ClassAcc_val = np.zeros([pNClasses])
  ClassCorrectCount_test = np.zeros([pNClasses])
  ClassAcc_test = np.zeros([pNClasses])
      
  # Determine correct train labels
  for i in range(len(y_train)):
      if y_train[i,0] == predictions_train[i]:
          #class_label = y_train[i,0]
          ClassCorrectCount_train[y_train[i,0]] +=1

  # Determine correct val labels
  for i in range(len(y_val)):
      if y_val[i,0] == predictions_val[i]:          
          ClassCorrectCount_val[y_val[i,0]] +=1
          
  # Determine correct test labels
  for i in range(len(y_test)):
      if y_test[i,0] == predictions_test[i]:          
          ClassCorrectCount_test[y_test[i,0]] +=1  

  TotalCorrect_train=np.sum(ClassCorrectCount_train)
  TotalCorrect_val= np.sum(ClassCorrectCount_val)
  TotalCorrect_test=np.sum(ClassCorrectCount_test)
    
  print('Total Correct Train: '+str(TotalCorrect_train))
  print('Total Acc Train: '+str(TotalCorrect_train/y_train.shape[0]))
   
  print('Total Correct val: '+str(TotalCorrect_val))
  print('Total Acc val: '+str(TotalCorrect_val/y_val.shape[0]))
    
  print('Total Correct test: '+str(TotalCorrect_test))
  print('Total Acc test: '+str(TotalCorrect_test/y_test.shape[0]))

  # Calculate Class Acc perc for each Class
  for i in range(pNClasses):
    ClassAcc_train[i] = ClassCorrectCount_train[i]/np.count_nonzero(y_train == i)   # Number of occurences of class i
    ClassAcc_val[i] = ClassCorrectCount_val[i]/np.count_nonzero(y_val == i)   # Number of occurences of class i
    ClassAcc_test[i] = ClassCorrectCount_test[i]/np.count_nonzero(y_test == i)   # Number of occurences of class i
  

  return ClassAcc_train, ClassAcc_val, ClassAcc_test
 





'''
******************************************************************************
*** Module: GeneXNet (Gene eXpression Network) ***

*** VISUALIZATION ***

******************************************************************************
'''



def Visualize_GeneXNet_ClassActivationMap_Heatmaps(pDataFileName, pDataGoogleDriveID, pNoGeneFilter=20):
  '''
  This function displays Heatmaps for GeneXNet ClassActivationMaps (Gene-CAM) for the different cancer tumors 
  The Gene-CAM is a gene localization map highlighting the important regions in the gene expressions which influenced the resulting tumor class prediction
  
    *** Input parameters: ***
    - pDataFileName: Gene Expression Heatmap Data FileName
    - pDataGoogleDriveID: Data File Google Drive ID
    - pNoGeneFilter: Number of Top Genes to filter when displaying Heatmap
    
    
  '''
  
    import seaborn as sns
    import pandas as pd 
    import matplotlib.pyplot as plt 
    import numpy as np
    from scipy import stats
    
    Init()                   # Check versions
    InitGoogleDriveAuth()    # Auth google drive
    
    FileName = pDataFileName
    FileGoogleDriveID = pDataGoogleDriveID
    
    ColorMaps =['PRGn','vlag','PiYG','PuOr','RdYlGn','BuPu','RdBu','RdGy','GnBu','PuRd','BrBG',]
    
    
    # Download GE Data from GoogleDrive 
    DownloadFiles = []
    DownloadFiles.append( dict( FileName=FileName , FileGoogleDriveID= FileGoogleDriveID ) ) 
    DownloadDataFiles(DownloadFiles)
    
    dfProjects = pd.read_excel(FileName)
    
    for index, data in dfProjects.iterrows():
    
        # Get FileName for Project
        ProjectID = data['ProjectID']
        ProjectTitle = data['ProjectTitleFull']
        ProjectSite = data['ProjectSite']      
        DataFile = 'TCGA_CaseData_TopGenes_'+ProjectID+'.xlsx'
        DataFileGoogleDriveID = data['GoogleDriveID'] 
        Flag = data['FlagHeatmap']
        if Flag=='OK':  
          # Download GE Data from GoogleDrive 
          DownloadFiles = []
          DownloadFiles.append( dict( FileName=DataFile , FileGoogleDriveID= DataFileGoogleDriveID ) ) 
          DownloadDataFiles(DownloadFiles)
    
    print()
    sns.set(font_scale=1.5)
    i=0
    for index, data in dfProjects.iterrows():
    
        # Get FileName for Project
        ProjectID = data['ProjectID']
        ProjectTitle = data['ProjectTitleFull']    
        ProjectSite = data['ProjectSite']      
        NoSamples = data['NoSamples']      
        DataFile = 'TCGA_CaseData_TopGenes_'+ProjectID+'.xlsx'
        Flag = data['FlagHeatmap']
        if Flag=='OK':  
          df = pd.read_excel(DataFile,index_col=0)
          df.rename(index=lambda x: 'Case-'+x[5:], inplace=True) 
          z = np.abs(stats.zscore(df, axis=1))
          df_filter = df[(z < 5).all(axis=1)]

          # Display Cluster Heatmap for Top Number of Genes 
          g=sns.clustermap(df.T.iloc[:pNoGeneFilter,:pNoGeneFilter],cmap=ColorMaps[i],standard_scale=1,  col_cluster=None , row_cluster=None, linewidths=0.2)

          g.ax_heatmap.set_title(ProjectTitle)
          g.ax_heatmap.set_xlabel('Tumor Samples (20/'+str(NoSamples)+')')
    
          i+=1
  
    
  return    




def Visualize_GeneXNet_MolecularClusters_Heatmaps(pDataFileName, pDataGoogleDriveID):
  '''
  This function displays Heatmaps for GeneXNet Molecular Clusters formed by intermediate gene expression feature maps
  Visualizing the molecular clustering helps in revealing the genomic relationships and high-level structures of gene expressions across multiple cancer tumor types
  
    *** Input parameters: ***
    - pDataFileName: Gene Expression Heatmap Data FileName
    - pDataGoogleDriveID: Data File Google Drive ID
    - pNoGeneFilter: Number of Top Genes to filter when displaying Heatmap
    
    
  '''
  
    import seaborn as sns
    import pandas as pd 
    import matplotlib.pyplot as plt 
    import numpy as np
    from scipy import stats
    import random
    
    
    Init()                   # Check versions
    InitGoogleDriveAuth()    # Auth google drive
    
    
    # Create a list of Site Names
    FileName = pDataFileName
    FileGoogleDriveID = pDataGoogleDriveID
    # Download GE Data from GoogleDrive 
    DownloadFiles = []
    DownloadFiles.append( dict( FileName=FileName , FileGoogleDriveID= FileGoogleDriveID ) ) 
    DownloadDataFiles(DownloadFiles)
    dfSites = pd.read_excel(FileName)
    SiteLabels=[]
    for index, data in dfSites.iterrows():
        # Get FileName for Project
        SiteLabels.append(data['ProjectPrimarySite'])
    
    
    # Create a list with 17 Group Clusters Labels 1 to 17
    GroupLabels =list(range(1,18))  
    
    GroupColorPal =sns.hls_palette(len(GroupLabels), l=0.4, s=0.8)
    SiteColorPal =sns.hls_palette(len(SiteLabels), l=0.4, s=0.8)
    # Create a shuffled copy of the color palettes
    GroupColorPalShuf = GroupColorPal.copy()
    SiteColorPalShuf = SiteColorPal.copy()
    random.shuffle(GroupColorPalShuf)
    random.shuffle(SiteColorPalShuf)
    
    # Create dict with color maps
    
    GroupColorMap = dict(zip(GroupLabels,GroupColorPalShuf))
    SiteColorMap = dict(zip(SiteLabels,SiteColorPalShuf))
    
    # ************ # Create List of Group Mapping **********
    # Load Sample counts for Groups and Sites from Excel
    FileName = 'ClusterMapProjectGroups.xlsx'
    FileGoogleDriveID = '1m6VDl-sH3uWqTh5H4ujhM7pxXSIIGJWU'
    # Download GE Data from GoogleDrive (Will only run if gLoadFilesFromGoogleDriveFlag is True)
    DownloadFiles = []
    DownloadFiles.append( dict( FileName=FileName , FileGoogleDriveID= FileGoogleDriveID ) ) 
    DownloadDataFiles(DownloadFiles)
    dfGroupSamples = pd.read_excel(FileName,sheet_name='Groups_200Samples_GroupMap')
    
    GroupMap = []
    for index, data in dfGroupSamples.iterrows():
        # Get FileName for Project
        GroupNo = data['GroupNo']
        GroupSampleCount = int(data['GroupSampleCount'])
        Groupperc = data['GroupPerc']
        # Add Samples for this group
        for k in range(GroupSampleCount):
          GroupMap.append(GroupNo)
    
    GroupColors =  [GroupColorMap[k] for k in GroupMap]
    
    
    # ************ # Create List of Site Mapping **********
    # Load Sample counts for Groups and Sites from Excel
    dfSiteSamples = pd.read_excel(FileName,sheet_name='Groups200Samples_SiteMap')
    
    SiteMap = []
    for index, data in dfSiteSamples.iterrows():
        SiteName = data['ProjectSite']
        SiteSampleCount = int(data['SiteSampleCount'])
        # Add Samples for this Site
        for k in range(SiteSampleCount):
          SiteMap.append(SiteName)
    
    SiteColors =  [SiteColorMap[k] for k in SiteMap]
      
    # *** GET DATA ***
    FileName = 'TCGA_CaseData_TopGenes_ALLDATA.xlsx'
    FileGoogleDriveID = '1rOXOBLRi9nnEmfnIWnTlGrkvW1n1y2e4'
    # Download GE Data from GoogleDrive 
    DownloadFiles = []
    DownloadFiles.append( dict( FileName=FileName , FileGoogleDriveID= FileGoogleDriveID ) ) 
    DownloadDataFiles(DownloadFiles)
    print('Loading data...')
    df = pd.read_excel(FileName,index_col=0)
    df.rename(index=lambda x: 'Case-'+x[5:], inplace=True) 
    
    z = np.abs(stats.zscore(df, axis=1))
    df_filter = df[(z < 10).all(axis=1)]
    
    df.fillna(value=0, inplace=True)
    
    # Remove columns with only one unique value such as a column with all zeros 
    df.drop(df.columns[df.nunique() == 1], axis=1, inplace=True  )  
    df.shape
    
    ColorMaps =['tab10','tab20','Set1','tab20','tab20b','tab20c']
   
    for i in range(len(ColorMaps)):
      print(ColorMaps[i])
 
      # Display Hierarchical Cluster Map 
      g=sns.clustermap(df.T.iloc[:,:9158],cmap=ColorMaps[i], standard_scale=0, col_cluster=True , row_cluster=None, linewidths=0, col_colors=[GroupColors,SiteColors])
    
  
      # Display Group color legends
      for i in range(len(GroupColorMap)):
        g.ax_col_dendrogram.bar(0, 0, color=GroupColorMap[i+1], label='Group '+str(GroupLabels[i]), linewidth=0) 
    
      l1 = g.ax_col_dendrogram.legend(title='Groups', loc="center", ncol=5, bbox_to_anchor=(0.47, 0.8))
    
      # Display Site color legends
      for Site in SiteColorMap.keys():
        g.ax_row_dendrogram.bar(0, 0, color=SiteColorMap[Site], label=Site, linewidth=0) 
    
      l2 = g.ax_row_dendrogram.legend(title='Sites', loc="center", ncol=5, bbox_to_anchor=(0.47, 0.8))
    
    
  return    




# *** Plotting Functions ***


def Plot_History(pNetworkHistory , plossmax, paccmin):


    plt.figure(figsize=(16,6))
    plt.subplot(1, 2, 1)  # No Rows, No Cols, index
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(pNetworkHistory['loss'])
    plt.plot(pNetworkHistory['val_loss'])
    plt.legend(['Training', 'Validation'])
    plt.ylim(top=plossmax, bottom=-0.02)  
    
    #plt.figure()
    plt.subplot(1, 2, 2) # No Rows, No Cols, index
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(pNetworkHistory['acc'])
    plt.plot(pNetworkHistory['val_acc'])
    plt.legend(['Training', 'Validation'], loc='lower right')  
    plt.ylim(top=1.01, bottom=paccmin)  
    return 

def plot_history_all(key,histories):
    # Plot all histories from Cross validation
    plt.figure(figsize=(16,10))

    for name, history in histories:
      val = plt.plot(history.epoch, history.history['val_'+key],
                     '--', label=name.title()+' Val')
      plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
               label=name.title()+' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_',' ').title())
    plt.legend()

    plt.xlim([0,max(history.epoch)])
    plt.show()



def plot_history_val(key,histories):
  # Plot validation only from all histories from Cross validation
  plt.figure(figsize=(16,10))
    
  for name, history in histories:
    val = plt.plot(history.epoch, history.history['val_'+key],
                   '--', label=name.title()+' Val')
 

  plt.xlabel('Epochs')
  plt.ylabel(key.replace('_',' ').title())
  plt.legend()

  plt.xlim([0,max(history.epoch)])
  plt.show()


def plot_history_train(key,histories):
  # Plot training only from all histories from Cross validation    
  plt.figure(figsize=(16,10))
    
  for name, history in histories:
    val = plt.plot(history.epoch, history.history[key], label=name.title()+' Train')
 

  plt.xlabel('Epochs')
  plt.ylabel(key.replace('_',' ').title())
  plt.legend()

  plt.xlim([0,max(history.epoch)])
  plt.show()

  


def CompareTrainingArchive_Plots(pSessionDataAll, pKey, pFilterSessionsFlag = False , pFilterSessions=[] , pFilterFoldsFlag = False, pFilterFolds=[] , pFigSize=(14,8), px_Axis_Limit=0, py_Axis_Limit=0, px_AxisTicks=[], py_AxisTicks=[],pShowModelBuild = True, pShowFinetuneParam=True, pShowComments=True ,  pShowTrainingFlag=True, pShowTrainingLabels=True, pShowValidationLabels=True,pShowModelTypeLabels=True, pSessionLabels = [], pPlotColors=[], pEpochsLimit = -1):
  '''
  This function visualizes and compares the training folds accross the different Training Sessions that have been previously recorded in a json file 
    *** Input parameters: ***
    - pSessionDataAll: List of dict which contains a the histiry of Training Sessions that have been previously recorded in a json file 
    
    - pShowTrainingFlag: Whether to display the training curve for all FoldGroupBatchs
    - pShowModelTypeLabels: Whether to use ModelType labels on Curves
    - pSessionLabels: Labels for the Sessions to override the default to be displayed which will allow us to control tthe labels 
    - pPlotColors
    
    
  '''
  
  NoTrainingSessions = len(pSessionDataAll)
  print ('Total number of training sessions: ' + str(NoTrainingSessions)+'\n')
  
  FoldBatchCounter =0
  Counter=0
  
  # Plot validation only from all histories from Cross validation
  plt.figure(figsize=pFigSize)
  key=pKey
  
 
  session_all=[]

  i=0
  while i < len(pSessionDataAll):  
    FoldBatchCounter += 1  # New Fold Batch

 
    # Updated 23 June 2019
    if 'NoFoldIterations' in pSessionDataAll[i]['TrainingParameters'].keys():
        NoFolds = pSessionDataAll[i]['TrainingParameters']['NoFoldIterations'] 
    else:
        NoFolds = pSessionDataAll[i]['SessionParameters']['NoFolds']

  
    if 'FoldIndexShift' in pSessionDataAll[i]['TrainingParameters'].keys():
        FoldIndexShift = pSessionDataAll[i]['TrainingParameters']['FoldIndexShift']
    else:
        FoldIndexShift = 0  # Normal

     
    FoldCounter = 0
    while FoldCounter < NoFolds:
      FoldCounter += 1  # increment fold counter within this fold batch  
      if (not pFilterSessionsFlag) or (pFilterSessionsFlag and FoldBatchCounter in pFilterSessions) : 
          if (not pFilterFoldsFlag) or (pFilterFoldsFlag and FoldCounter+FoldIndexShift in pFilterFolds):  
          Counter+=1  
          
          # get Plot Color if given as parameter
          PlotColor=''
          if len(pPlotColors)!=0:
              PlotColor = pPlotColors[Counter-1]
          
          PlotLabel=''
          # Check if we have labels for the curves
          if len(pSessionLabels)!=0:
              if pSessionLabels[Counter-1]!='':
                  PlotLabel = pSessionLabels[Counter-1]                 

          if pShowModelTypeLabels:
              PlotLabel = pSessionDataAll[i]['SessionParameters']['ModelBuildParameters']['Model'] 

          # Use default Session and Fold No
          if PlotLabel=='':
              PlotLabel = 'Session '+ str(FoldBatchCounter)+ ' Fold ' + str(FoldCounter+FoldIndexShift)


          history = pSessionDataAll[i]['TrainingHistory']

          if pShowTrainingFlag:  
              if pShowTrainingLabels: 
                  if pShowValidationLabels:
                     PlotLabelTraining = PlotLabel+' Training'
                  else:
                     PlotLabelTraining = PlotLabel 
                  
              else:
                  PlotLabelTraining = '_nolegend_'
              
              if PlotColor!='':
                  train = plt.plot(history[key][:pEpochsLimit], label=PlotLabelTraining, linestyle='dashed', color=PlotColor)         
              else:
                  train = plt.plot(history[key][:pEpochsLimit], label=PlotLabelTraining, linestyle='dashed')         


          if pShowValidationLabels:    
              if pShowTrainingLabels: 
                  PlotLabelValidation = PlotLabel+' Validation'
              else:
                  PlotLabelValidation = PlotLabel
          else:
              PlotLabelValidation = '_nolegend_'

          if PlotColor!='':
              val = plt.plot(history['val_'+key][:pEpochsLimit], label=PlotLabelValidation, color=PlotColor)         
          else:
              val = plt.plot(history['val_'+key][:pEpochsLimit], label=PlotLabelValidation)         
          
                              
      
      i +=1  # move to the next training session
      # End while FoldCounter < NoFolds      

    # Fold batch has been all read  
 
    # Loop to next fold batch
    # End while i < len(pSessionDataAll)
    
 
  plt.xlabel('Epochs')
  plt.ylabel(key.replace('_',' ').title())
  if pKey=='loss':
      plt.ylabel('Crossentropy Loss')
  
  if px_Axis_Limit!=0:
      plt.xlim(right=px_Axis_Limit)
      plt.xlim(left=0)
      
  if py_Axis_Limit!=0:
      plt.ylim(top=py_Axis_Limit)
      plt.ylim(bottom=-0.01)

  if pShowValidationLabels or pShowTrainingLabels:
      plt.legend() 
  
  plt.grid(True, color = 'silver', linestyle='dotted', linewidth=1)   
  if len(px_AxisTicks)!=0:
      axis=plt.axes()
      axis.set_xticks(px_AxisTicks)

  if len(py_AxisTicks)!=0:
      axis=plt.axes()
      axis.set_yticks(py_AxisTicks)

  plt.show()
  
    
  return    




def CompareTrainingArchive_ROC(pSessionDataAll, pKey='test', pFilterSessionsFlag = False , pFilterSessions=[] , pFilterFoldsFlag = False, pFilterFolds=[] , pFigSize=(14,8), px_Axis_Limit=0, py_Axis_Limit=0, pAxisTicks=[], pShowModelBuild = True, pShowFinetuneParam=True, pShowComments=True , pShowModelTypeLabels=True, pShowLabels=True, pSessionLabels = [], pPlotColors=[]):
  '''
  This function compares the ROC Curves for folds accross the different Training Sessions that have been previously recorded in a json file 
    *** Input parameters: ***
    - pSessionDataAll: List of dict which contains a the histiry of Training Sessions that have been previously recorded in a json file 
  '''
  
  NoTrainingSessions = len(pSessionDataAll)
  print ('Total number of training sessions: ' + str(NoTrainingSessions)+'\n')
  
  FoldBatchCounter =0
  Counter=0
  
  # Plot validation only from all histories from Cross validation
  plt.figure(figsize=pFigSize)
  key=pKey
  
 
  session_all=[]
  
  
  i=0
  while i < len(pSessionDataAll):  
    FoldBatchCounter += 1  # New Fold Batch

    if 'NoFoldIterations' in pSessionDataAll[i]['TrainingParameters'].keys():
        NoFolds = pSessionDataAll[i]['TrainingParameters']['NoFoldIterations'] 
    else:
        NoFolds = pSessionDataAll[i]['SessionParameters']['NoFolds']

    if 'FoldIndexShift' in pSessionDataAll[i]['TrainingParameters'].keys():
        FoldIndexShift = pSessionDataAll[i]['TrainingParameters']['FoldIndexShift']
    else:
        FoldIndexShift = 0  # Normal

       
    FoldCounter = 0
    while FoldCounter < NoFolds:
      FoldCounter += 1  # increment fold counter within this fold batch  
      if (not pFilterSessionsFlag) or (pFilterSessionsFlag and FoldBatchCounter in pFilterSessions) : 
        if (not pFilterFoldsFlag) or (pFilterFoldsFlag and FoldCounter+FoldIndexShift in pFilterFolds):  
                        
            Counter+=1

            # get Plot Color if given as parameter
            PlotColor=''
            if len(pPlotColors)!=0:
                PlotColor = pPlotColors[Counter-1]

       
            PlotLabel=''
            # Check if we have labels for the curves
            if len(pSessionLabels)!=0:
                if pSessionLabels[Counter-1]!='':
                    PlotLabel = pSessionLabels[Counter-1]                 
    
            if pShowModelTypeLabels:
                PlotLabel = pSessionDataAll[i]['SessionParameters']['ModelBuildParameters']['Model'] 
    
            # Use default Session and Fold No
            if PlotLabel=='':
                PlotLabel = 'ROC '+'Session '+ str(FoldBatchCounter)+ ' Fold ' + str(FoldCounter+FoldIndexShift)+' '+pKey+' AUC=%0.3f)' % Predictions[pKey]['ROC_AUC']

            Predictions = pSessionDataAll[i]['Predictions']    
            if len(Predictions)!=0:  
                if PlotColor!='':
                    plt.plot(Predictions[pKey]['ROC_FPR'], Predictions[pKey]['ROC_TPR'],color=PlotColor, label=PlotLabel)
                else:
                    plt.plot(Predictions[pKey]['ROC_FPR'], Predictions[pKey]['ROC_TPR'],label=PlotLabel)
          
       
      i +=1  # move to the next training session
      # End while FoldCounter < NoFolds      

    # Fold batch has been all read    
    # End while i < len(pSessionDataAll)
    
   if pShowLabels:
      plt.legend(loc="lower right")          
    
  plt.plot([0, 1], [0, 1], color='black',  linestyle='--')
  #plt.xlim([0.0, 1.0])
  plt.xlim([-0.05, 1.05])
  plt.ylim([-0.05, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('ROC')
    
    
  if px_Axis_Limit!=0:
      plt.xlim(right=px_Axis_Limit)
      plt.xlim(left=-0.01)
      
  if py_Axis_Limit!=0:                    
      plt.ylim(top=py_Axis_Limit)
      plt.ylim(bottom=-0.01)
    
    
  plt.grid(True, color = 'silver', linestyle='--', linewidth=1) 
    
    
  if len(pAxisTicks)!=0:
      axis=plt.axes()
      axis.set_xticks(pAxisTicks)
      axis.set_yticks(pAxisTicks)
     
  plt.show()
     
    
  return    






