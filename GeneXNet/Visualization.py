"""
******************************************************************************
*** Package: GeneXNet (Gene eXpression Network) ***
*** Module: Visualization ***

*** VISUALIZATION ***

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

print ('Copyright (c) 2018-2020, Tarek Khorshed (Tarek_Khorshed@aucegypt.edu)')   

'''
******************************************************************************
*** Module: GeneXNet (Gene eXpression Network) ***

*** VISUALIZATION ***

******************************************************************************
'''


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






