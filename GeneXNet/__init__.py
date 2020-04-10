"""
******************************************************************************
*** Package: GeneXNet (Gene eXpression Network) ***

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
print ('Initializing GeneXNet Package...')  
print ('Copyright (c) 2020, Tarek Khorshed (Tarek_Khorshed@aucegypt.edu)')

#*** GLOBAL VARIABLES ***

# Platform variables
'''
- The platform that the Package will be executing from. 
  GoogleColab: GoogleColab notebook (Default)
  LocalCode: Local python code which could be running in Spyder, Cmdline or Visual Studio Code or others
  LocalNotebook: Local Jupyter Notebook 
'''
cPlatform_GoogleColab ='GoogleColab'
cPlatform_LocalCode ='LocalCode'
cPlatform_LocalNotebook ='LocalNotebook'
gPlatform = cPlatform_GoogleColab    # Default

'''
 - Flag to indicate whether we are running a Pipeline Batch mode or normal interactive mode
'''
gPipelineModeFlag = False    # Default
gPipelineBatchNo = ''    # Deafult - BatchNo suffix string which is used for creating the filenames for training history and weights

'''
 - Flag to indicate whether to Download files from google drive first before loading or load directly from local drive
 - This helps in running the code in spyder or local notebooks where flag can be set to false and files will be loaded from local folder
 - For GoogleClab it will always be true which is the default   
'''
gLoadFilesFromGoogleDriveFlag = True  # Default   
gUploadFilesToGoogleDriveFlag = True  # Default   

# GoogleDrive Path Variables
gGoogleDriveAuth = None       # Used for setting the gAuth object when authenticating for Google Drive
gGoogleDrive_DataFoldsFolderID = ''
gGoogleDrive_SessionDataFolderID = ''
gGoogleDrive_ExportDataFolderID = ''
gGoogleDrive_WeightDataFolderID = ''
gGoogleDrive_ModelDataFolderID = ''

# Matplot Plot Style
gMatPlotLibStyle = 'seaborn'
