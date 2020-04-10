"""
******************************************************************************
*** Package: GeneXNet (Gene eXpression Network) ***
*** Module: SessionIO ***

*** SESSION INPUT/OUTPUT MANAGEMENT ***

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

print ('Copyright (c) 2020, Tarek Khorshed (Tarek_Khorshed@aucegypt.edu)')   


'''
******************************************************************************
*** Module: GeneXNet (Gene eXpression Network) ***

*** SESSION INPUT/OUTPUT MANAGEMENT ***

******************************************************************************
'''

import numpy as np
import seaborn as sns
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
 




