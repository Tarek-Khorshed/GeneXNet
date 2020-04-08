"""
******************************************************************************
*** Package: GeneXNet (Gene eXpression Network) ***
*** Module: Training ***

*** MODEL TRAINING ***

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

*** MODEL TRAINING ***

******************************************************************************
'''

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
from time import time
import os
import math

import uuid
import datetime
import json
import copy
from collections import OrderedDict

import psutil


  
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
 

