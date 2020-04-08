"""
******************************************************************************
*** Package: GeneXNet (Gene eXpression Network) ***
*** Module: Models ***

*** MODEL BUILDING ***

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

*** MODEL BUILDING ***

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


