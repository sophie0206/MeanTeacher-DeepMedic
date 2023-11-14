# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division
import numpy as np
import random
from math import ceil
from collections import OrderedDict

import tensorflow as tf

from deepmedicUDA.neuralnet.pathwayTypes import PathwayTypes as pt
from deepmedicUDA.neuralnet.pathways import NormalPathway, SubsampledPathway, FcPathway, DDPathway, getMiddlePartOfFms
from deepmedicUDA.neuralnet.layers import SoftmaxLayer, ConvLayer


from deepmedicUDA.neuralnet.utils import calcRecFieldFromKernDimListPerLayerWhenStrides1

######### Helper functions used in this module ########

def padImageWithMirroring(inputImage, voxelsPerDimToPad) :
    # inputImage shape: [batchSize, #channels#, r, c, z]
    # inputImageDimensions : [ batchSize, #channels, dim r, dim c, dim z ] of inputImage
    # voxelsPerDimToPad shape: [ num o voxels in r-dim to add, ...c-dim, ...z-dim ]
    # If voxelsPerDimToPad is odd, 1 more voxel is added to the right side.
    # r-axis
    assert np.all(voxelsPerDimToPad) >= 0
    padLeft = int(voxelsPerDimToPad[0] // 2); padRight = int((voxelsPerDimToPad[0] + 1) // 2);
    paddedImage = tf.concat([inputImage[:, :, int(voxelsPerDimToPad[0] // 2) - 1::-1 , :, :], inputImage], axis=2) if padLeft > 0 else inputImage
    paddedImage = tf.concat([paddedImage, paddedImage[ :, :, -1:-1 - int((voxelsPerDimToPad[0] + 1) // 2):-1, :, :]], axis=2) if padRight > 0 else paddedImage
    # c-axis
    padLeft = int(voxelsPerDimToPad[1] // 2); padRight = int((voxelsPerDimToPad[1] + 1) // 2);
    paddedImage = tf.concat([paddedImage[:, :, :, padLeft - 1::-1 , :], paddedImage], axis=3) if padLeft > 0 else paddedImage
    paddedImage = tf.concat([paddedImage, paddedImage[:, :, :, -1:-1 - padRight:-1, :]], axis=3) if padRight > 0 else paddedImage
    # z-axis
    padLeft = int(voxelsPerDimToPad[2] // 2); padRight = int((voxelsPerDimToPad[2] + 1) // 2)
    paddedImage = tf.concat([paddedImage[:, :, :, :, padLeft - 1::-1 ], paddedImage], axis=4) if padLeft > 0 else paddedImage
    paddedImage = tf.concat([paddedImage, paddedImage[:, :, :, :, -1:-1 - padRight:-1]], axis=4) if padRight > 0 else paddedImage
    
    return paddedImage


##################################################
##################################################
################ THE CNN CLASS ###################
##################################################
##################################################

class Cnn3d(object):
    def __init__(self):
        
        self.cnnModelName = None
        
        self.pathways = [] # There should be only 1 normal and only one FC pathway. Eg, see self.getFcPathway()
        self.numSubsPaths = 0
        
        self.finalTargetLayer = ""
        
        self.num_classes = None
        
        #=====================================
        self.recFieldCnn = ""
                
        self.batchSize = {"train": "", "val": "", "test": ""}
        
        # self.patchesToTrainPerImagePart = ""
        self.nkerns = ""  # number of feature maps.
        self.nkernsSubsampled = ""
        
        # Fully Connected Layers
        self.kernelDimensionsFirstFcLayer = ""
        
        # Residual Learning
        self.indicesOfLayersToConnectResidualsInOutput = ""
        
        # Lower rank convolutional layers
        self.indicesOfLowerRankLayersPerPathway = ""
        self.ranksOfLowerRankLayersForEachPathway = ""
        
        #======= Input tensors X. Placeholders OR given tensors =======
        # Symbolic variables, which stand for the input. Will be loaded by the compiled trainining/val/test function. Can also be pre-set by an existing tensor if required in future extensions.
        self._inp_x = { 'train': {},
                        'val': {},
                        'test': {} }
        
        
        #======= Output tensors Y_GT ========
        # For each targetLayer, I should be placing a y_gt placeholder/feed, by calls to finalTargetLayer.get_output_gt_tensor_feed()
        self._output_gt_tensor_feeds = {'train': {},'val': {} }
        ##==========================the Gt labels output of DD, needed when computing DDcost======##
        self._DDoutput_gt_tensor_feeds = {'train': {},'val': {} }
        ##=========================================================================================##
        ######## These entries are setup in the setup_train/val/test functions here ############
        self._ops_main = { 'train': {} , 'val': {}, 'test': {} }
        self._feeds_main = { 'train': {} , 'val': {}, 'test': {} }

    
    def getNumSubsPathways(self):
        count = 0
        for pathway in self.pathways :
            if pathway.pType() ==  pt.SUBS :
                count += 1
        return count
    
    def getNumPathwaysThatRequireInput(self):
        count = 0
        for pathway in self.pathways :
            if pathway.pType() != pt.FC :
                count += 1
        return count
    
    def getFcPathway(self):
        for pathway in self.pathways :
            if pathway.pType() == pt.FC :
                return pathway
        return None

    ##======================Add getDDPathway =======================##
    def getDDPathway(self):
        for pathway in self.pathways :
            if pathway.pType() == pt.DD :
                return pathway
        log.print3("No DD Pathway!!!!!!!!!!")
        return None;


    

        
    # for inference with batch-normalization. Every training batch, this is called to update an internal matrix of each layer, with the last mus and vars, so that I can compute the rolling average for inference.
    def updateMatricesOfBnMovingAvForInference(self, sessionTf) :
        self._updateMatricesOfBnMovingAvForInference(sessionTf)
        
    def _updateMatricesOfBnMovingAvForInference(self, sessionTf):
        for pathway in self.pathways :
            for layer in pathway.getLayers() :
                layer.updateMatricesOfBnMovingAvForInference(sessionTf)  # Will do nothing if no BN.
                    
    def _getUpdatesForBnRollingAverage(self) :
        # These are not the variables of the normalization of the FMs' distributions that are optimized during training. These are only the Mu and Stds that are used during inference,
        # ... and here we update the sharedVariable which is used "from the outside during do_training()" to update the rolling-average-matrix for inference. Do for all layers.
        updatesForBnRollingAverage = []
        for pathway in self.pathways :
            for layer in pathway.getLayers() :
                updatesForBnRollingAverage.extend(layer.getUpdatesForBnRollingAverage())
        return updatesForBnRollingAverage
    
    def get_trainable_params(self, log, indicesOfLayersPerPathwayTypeToFreeze):
        # Called from Trainer.
        paramsToOptDuringTraining = []  # Ws and Bs
        for pathway in self.pathways :
            for layer_i in range(0, len(pathway.getLayers())) :
                if pathway.pType() == pt.DD:
                    continue
                if layer_i not in indicesOfLayersPerPathwayTypeToFreeze[ pathway.pType() ] :
                    paramsToOptDuringTraining = paramsToOptDuringTraining + pathway.getLayer(layer_i).getTrainableParams()
                else : # Layer will be held fixed. Notice that Batch Norm parameters are still learnt.
                    log.print3("WARN: [Pathway_" + str(pathway.getStringType()) + "] The weights of [Layer-"+str(layer_i)+"] will NOT be trained as specified (index, first layer is 0).")
        return paramsToOptDuringTraining
    
    def _get_L1_cost(self) :
        L1 = 0
        for pathway in self.pathways :
            for layer in pathway.getLayers() :
                L1 += layer._get_L1_cost()
        return L1
    
    def _get_L2_cost(self) :
        L2_sqr = 0
        for pathway in self.pathways :
            for layer in pathway.getLayers() :    
                L2_sqr += layer._get_L2_cost()
        return L2_sqr
    
    def get_main_ops(self, str_train_val_test):
     # str_train_val_test: "train", "val" or "test"
        return self._ops_main[str_train_val_test]
    
    def get_main_feeds(self, str_train_val_test):
        return self._feeds_main[str_train_val_test]
    
    
    def setup_ops_n_feeds_to_train(self, log, total_cost, updates_of_params_wrt_total_cost) :
        log.print3("...Building the training function...")
        
        y_gt = self._output_gt_tensor_feeds['train']['y_gt']
        yDD_gt = self._DDoutput_gt_tensor_feeds['train']['y_gt']
        
        #================BATCH NORMALIZATION ROLLING AVERAGE UPDATES======================
        updates = updates_of_params_wrt_total_cost + self._getUpdatesForBnRollingAverage()
        updates_grouped_op = tf.group( *updates ) # this op returns no output when run.
        
        #======================== Collecting ops and feeds =================
        log.print3("...Collecting ops and feeds for training...")
        
        self._ops_main['train']['cost'] = total_cost
        self._ops_main['train']['list_rp_rn_tp_tn'] = self.finalTargetLayer.getRpRnTpTnForTrain0OrVal1(y_gt, 0)
        ##===========================================================##
        self._ops_main['train']['DDlist_rp_rn_tp_tn'] = self.DDfinalTargetLayer.getRpRnTpTnForTrain0OrVal1(yDD_gt, 0)
        ##===========================================================##
        self._ops_main['train']['updates_grouped_op'] = updates_grouped_op
        
        self._feeds_main['train']['x'] = self._inp_x['train']['x']
        for subpath_i in range(self.numSubsPaths) : # if there are subsampled paths...
            self._feeds_main['train']['x_sub_'+str(subpath_i)] = self._inp_x['train']['x_sub_'+str(subpath_i)]
        self._feeds_main['train']['y_gt'] = y_gt
        self._feeds_main['train']['yDD_gt'] = yDD_gt

        
        log.print3("Done.")
    
   

    def setup_ops_n_feeds_to_val(self, log) :
        log.print3("...Building the validation function...")
        
        y_gt = self._output_gt_tensor_feeds['val']['y_gt']
        
        log.print3("...Collecting ops and feeds for validation...")
        
        self._ops_main['val'] = {}
        self._ops_main['val']['list_rp_rn_tp_tn'] = self.finalTargetLayer.getRpRnTpTnForTrain0OrVal1(y_gt, 1)
        
        self._feeds_main['val'] = {}
        self._feeds_main['val']['x'] = self._inp_x['val']['x']
        for subpath_i in range(self.numSubsPaths) : # if there are subsampled paths...
            self._feeds_main['val']['x_sub_'+str(subpath_i)] = self._inp_x['val']['x_sub_'+str(subpath_i)]
        self._feeds_main['val']['y_gt'] = y_gt
        
        log.print3("Done.")
        
        
    def setup_ops_n_feeds_to_test(self, log, indices_fms_per_pathtype_per_layer_to_save=None) :
        log.print3("...Building the function for testing and visualisation of FMs...")
        
        listToReturnWithAllTheFmActivationsPerLayer = []
        if indices_fms_per_pathtype_per_layer_to_save is not None:
            for pathway in self.pathways :
                if pathway.pType() == pt.DD:
                    continue;
                indicesOfFmsToVisualisePerLayerOfCertainPathway = indices_fms_per_pathtype_per_layer_to_save[ pathway.pType() ]
                if indicesOfFmsToVisualisePerLayerOfCertainPathway != [] :
                    layers = pathway.getLayers()
                    for layer_i in range(len(layers)) :  # each layer that this pathway/fc has.
                        indicesOfFmsToExtractFromThisLayer = indicesOfFmsToVisualisePerLayerOfCertainPathway[layer_i]
                        if len(indicesOfFmsToExtractFromThisLayer) > 0: #if no FMs are to be taken, this should be []
                            listToReturnWithAllTheFmActivationsPerLayer.append( layers[layer_i].fmsActivations(indicesOfFmsToExtractFromThisLayer) )
        
        log.print3("...Collecting ops and feeds for testing...")
        
        self._ops_main['test'] = {}
        self._ops_main['test']['list_of_fms_per_layer'] = listToReturnWithAllTheFmActivationsPerLayer
        self._ops_main['test']['pred_probs'] = self.finalTargetLayer.predictionProbabilities()
        
        self._feeds_main['test'] = {}
        self._feeds_main['test']['x'] = self._inp_x['test']['x']
        for subpath_i in range(self.numSubsPaths) : # if there are subsampled paths...
            self._feeds_main['test']['x_sub_'+str(subpath_i)] = self._inp_x['test']['x_sub_'+str(subpath_i)]
        
        log.print3("Done.")
        
        
        
        
    def _setupInputXTensors(self):
        self._inp_x['train']['x'] = tf.placeholder(dtype="float32", shape=[None, None, None, None, None], name="inp_x_train")
        self._inp_x['val']['x'] = tf.placeholder(dtype="float32", shape=[None, None, None, None, None], name="inp_x_val")
        self._inp_x['test']['x'] = tf.placeholder(dtype="float32", shape=[None, None, None, None, None], name="inp_x_test")
        for subpath_i in range(self.numSubsPaths) : # if there are subsampled paths...
            self._inp_x['train']['x_sub_'+str(subpath_i)] = tf.placeholder(dtype="float32", shape=[None, None, None, None, None], name="inp_x_sub_"+str(subpath_i)+"_train")
            self._inp_x['val']['x_sub_'+str(subpath_i)] = tf.placeholder(dtype="float32", shape=[None, None, None, None, None], name="inp_x_sub_"+str(subpath_i)+"_val")
            self._inp_x['test']['x_sub_'+str(subpath_i)] = tf.placeholder(dtype="float32", shape=[None, None, None, None, None], name="inp_x_sub_"+str(subpath_i)+"_test")
            
        
    def _setupInputXTensorsFromGivenArgs(self, givenInputTensorNormTrain, givenInputTensorNormVal, givenInputTensorNormTest,
                                         givenListInputTensorPerSubsTrain, givenListInputTensorPerSubsVal, givenListInputTensorPerSubsTest):
        self._inp_x['train']['x'] = givenInputTensorNormTrain
        self._inp_x['val']['x'] = givenInputTensorNormVal
        self._inp_x['test']['x'] = givenInputTensorNormTest
        for subpath_i in range(self.numSubsPaths):
            self._inp_x['train']['x_sub_'+str(subpath_i)] = givenListInputTensorPerSubsTrain[subpath_i]
            self._inp_x['val']['x_sub_'+str(subpath_i)] = givenListInputTensorPerSubsVal[subpath_i]
            self._inp_x['test']['x_sub_'+str(subpath_i)] = givenListInputTensorPerSubsTest[subpath_i]
        
        
    def _getClassificationLayer(self):
        return SoftmaxLayer()
        
        
    def make_cnn_model( self,
                        log,
                        cnnModelName,
                        #=== Model Parameters ===
                        numberOfOutputClasses,
                        numberOfImageChannelsPath1,
                        numberOfImageChannelsPath2,
                        
                        #=== Normal Pathway ===
                        nkerns,
                        kernelDimensions,
                        #=== Subsampled Pathway ===
                        # THESE NEXT TWO, ALONG WITH THE ONES FOR FC, COULD BE PUT IN ONE STRUCTURE WITH NORMAL, EG LIKE kerns = [ [kernsNorm], [kernsSub], [kernsFc]]
                        nkernsSubsampled, # Used to control if secondary pathways: [] if no secondary pathways. Now its the "factors"
                        kernelDimensionsSubsampled,
                        subsampleFactorsPerSubPath, # Controls how many pathways: [] if no secondary pathways. Else, List of lists. One sublist per secondary pathway. Each sublist has 3 ints, the rcz subsampling factors.
                        #=== FC Layers ===
                        fcLayersFMs,
                        kernelDimensionsFirstFcLayer,
                        softmaxTemperature,
                        
                        #=== Other Architectural params ===
                        activationFunc,
                        #---Residual Connections----
                        indicesOfLayersToConnectResidualsInOutput,
                        #--Lower Rank Layer Per Pathway---
                        indicesOfLowerRankLayersPerPathway,
                        ranksOfLowerRankLayersForEachPathway,
                        #---Pooling---
                        maxPoolingParamsStructure,
                        #--- Skip Connections --- #Deprecated, not used/supported
                        convLayersToConnectToFirstFcForMultiscaleFromAllLayerTypes,
                        
                        #===Size of Image Segments ===
                        imagePartDimensionsTraining , # a list , like[37,37,37]
                        imagePartDimensionsValidation,
                        imagePartDimensionsTesting,
                        
                        #=== Batch Sizes ===
                        batch_size_train,
                        batch_size_validation,
                        batch_size_testing,

                        
                        #=== Others ===
                        # Dropout
                        dropoutRatesForAllPathways,  # list of sublists, one for each pathway. Each either empty or full with the dropout rates of all the layers in the path.
                        # Initialization
                        convWInitMethod,
                        # Batch Normalization
                        applyBnToInputOfPathways,  # one Boolean flag per pathway type. Placeholder for the FC pathway.
                        movingAvForBnOverXBatches,
                        
                        ):
        
        self.cnnModelName = cnnModelName
        
        # ============= Model Parameters Passed as arguments ================
        self.num_classes = numberOfOutputClasses
        self.DDnum_classes = 2
        self.numberOfImageChannelsPath1 = numberOfImageChannelsPath1
        self.numberOfImageChannelsPath2 = numberOfImageChannelsPath2
        # === Architecture ===
        self.nkerns = nkerns  # Useless?
        self.nkernsSubsampled = nkernsSubsampled  # Useless?
        self.numSubsPaths = len(subsampleFactorsPerSubPath) # do I want this as attribute? Or function is ok?
        
        # fcLayersFMs???
        self.kernelDimensionsFirstFcLayer = kernelDimensionsFirstFcLayer
        
        # == Other Architectural Params ==
        self.indicesOfLayersToConnectResidualsInOutput = indicesOfLayersToConnectResidualsInOutput
        self.indicesOfLowerRankLayersPerPathway = indicesOfLowerRankLayersPerPathway
        # pooling?

        # == Batch Sizes ==
        self.batchSize = {"train": batch_size_train, "val": batch_size_validation, "test": batch_size_testing}
        ##============================================================================================#
        self.batchSize_seg = {"train": (batch_size_train//2), "val": (batch_size_validation//2), "test": (batch_size_testing//2)} # not sure about the val batch size
        self.batchSize_adv = {"train": (batch_size_train//2), "val": (batch_size_validation//2), "test": (batch_size_testing//2)}
        ##============================================================================================#
        # == Others ==
        self.dropoutRatesForAllPathways = dropoutRatesForAllPathways
        
        # ======== Calculated Attributes =========
        #This recField CNN should in future be calculated with all non-secondary pathways, ie normal+fc. Use another variable for pathway.recField.
        self.recFieldCnn = calcRecFieldFromKernDimListPerLayerWhenStrides1(kernelDimensions)
        
        #==============================
        rng = np.random.RandomState(23455)
        
        ######################
        # BUILD ACTUAL MODEL #
        ######################
        log.print3("...Building the CNN model...")
        
        # Symbolic variables, which stand for the input. Will be loaded by the compiled trainining/val/test function.
        # >>> I should have an input argument. Which, if given None, the below placeholders are created.
        
        if True: # Not given input tensors as arguments
            self._setupInputXTensors()
        else: # Inputs given as argument tensors. Eg in adv from discr or from batcher.
            self._setupInputXTensorsFromGivenArgs(1,2,3,4,5,6) # Placeholder. Todo: Replace with normal arguments, when input tensor is given. Eg adversarial G.

        
        #=======================Make the NORMAL PATHWAY of the CNN=======================
        thisPathway = NormalPathway()
        self.pathways.append(thisPathway)
        thisPathwayType = thisPathway.pType()

        
        inputToPathwayTrain = self._inp_x['train']['x']
        inputToPathwayVal = self._inp_x['val']['x']
        inputToPathwayTest = self._inp_x['test']['x']
        inputToPathwayShapeTrain = [self.batchSize["train"], numberOfImageChannelsPath1] + imagePartDimensionsTraining
        inputToPathwayShapeVal = [self.batchSize["val"], numberOfImageChannelsPath1] + imagePartDimensionsValidation
        inputToPathwayShapeTest = [self.batchSize["test"], numberOfImageChannelsPath1] + imagePartDimensionsTesting
        
        thisPathWayNKerns = nkerns
        thisPathWayKernelDimensions = kernelDimensions
        
        thisPathwayNumOfLayers = len(thisPathWayNKerns)

        thisPathwayUseBnPerLayer = [movingAvForBnOverXBatches > 0] * thisPathwayNumOfLayers
        thisPathwayUseBnPerLayer[0] = applyBnToInputOfPathways[thisPathwayType] if movingAvForBnOverXBatches > 0 else False  # For the 1st layer, ask specific flag.
        
        thisPathwayActivFuncPerLayer = [activationFunc] * thisPathwayNumOfLayers
        thisPathwayActivFuncPerLayer[0] = "linear" if thisPathwayType != pt.FC else activationFunc  # To not apply activation on raw input. -1 is linear activation.
        
        thisPathway.makeLayersOfThisPathwayAndReturnDimensionsOfOutputFM(log,
                                                                         inputToPathwayTrain,
                                                                         inputToPathwayVal,
                                                                         inputToPathwayTest,
                                                                         inputToPathwayShapeTrain,
                                                                         inputToPathwayShapeVal,
                                                                         inputToPathwayShapeTest,
                                                                         
                                                                         thisPathWayNKerns,
                                                                         thisPathWayKernelDimensions,
                                                                         
                                                                         convWInitMethod,
                                                                         thisPathwayUseBnPerLayer,
                                                                         movingAvForBnOverXBatches,
                                                                         thisPathwayActivFuncPerLayer,
                            
self.batchSize_seg,
                                                                         self.batchSize_adv,
                                                                         dropoutRatesForAllPathways[thisPathwayType],
                                                                         
                                                                         maxPoolingParamsStructure[thisPathwayType],
                                                                         
                                                                         indicesOfLowerRankLayersPerPathway[thisPathwayType],
                                                                         ranksOfLowerRankLayersForEachPathway[thisPathwayType],
                                                                         indicesOfLayersToConnectResidualsInOutput[thisPathwayType]                                                     )
        # Skip connections to end of pathway.
        thisPathway.makeMultiscaleConnectionsForLayerType(convLayersToConnectToFirstFcForMultiscaleFromAllLayerTypes[thisPathwayType])
        
        [dimsOfOutputFrom1stPathwayTrain, dimsOfOutputFrom1stPathwayVal, dimsOfOutputFrom1stPathwayTest] = thisPathway.getShapeOfOutput()
        
        ##=============================Get the DDOutput from Normal Pathway and the Shapes for later upsampling===========================================================================================##
        [OutputForDDFrom1stPathwayTrain, 
        OutputForDDFrom1stPathwayVal, 
        OutputForDDFrom1stPathwayTest] = thisPathway.getOutputDD()

        [dimsOfOutputForDDFrom1stPathwayTrain, 
        dimsOfOutputForDDFrom1stPathwayVal, 
        dimsOfOutputForDDFrom1stPathwayTest] = thisPathway.getShapeOfOutputDD()
        ##======================================================================================================================================================##


        #=======================Make the SUBSAMPLED PATHWAYs of the CNN========================================
        for subpath_i in range(self.numSubsPaths) :
            thisPathway = SubsampledPathway(subsampleFactorsPerSubPath[subpath_i])
            self.pathways.append(thisPathway) # There will be at least an entry as a secondary pathway. But it won't have any layers if it was not actually used.
            thisPathwayType = thisPathway.pType()
            
            inputToPathwayTrain = self._inp_x['train']['x_sub_'+str(subpath_i)]
            inputToPathwayVal = self._inp_x['val']['x_sub_'+str(subpath_i)]
            inputToPathwayTest = self._inp_x['test']['x_sub_'+str(subpath_i)]
            
            thisPathWayNKerns = nkernsSubsampled[subpath_i]
            thisPathWayKernelDimensions = kernelDimensionsSubsampled
            
            thisPathwayNumOfLayers = len(thisPathWayNKerns)
            thisPathwayUseBnPerLayer = [movingAvForBnOverXBatches > 0] * thisPathwayNumOfLayers
            thisPathwayUseBnPerLayer[0] = applyBnToInputOfPathways[thisPathwayType] if movingAvForBnOverXBatches > 0 else False  # For the 1st layer, ask specific flag.
            
            thisPathwayActivFuncPerLayer = [activationFunc] * thisPathwayNumOfLayers
            thisPathwayActivFuncPerLayer[0] = "linear" if thisPathwayType != pt.FC else activationFunc  # To not apply activation on raw input. -1 is linear activation.
            
            inputToPathwayShapeTrain = [self.batchSize["train"], numberOfImageChannelsPath2] + thisPathway.calcInputRczDimsToProduceOutputFmsOfCompatibleDims(thisPathWayKernelDimensions, dimsOfOutputFrom1stPathwayTrain);
            inputToPathwayShapeVal = [self.batchSize["val"], numberOfImageChannelsPath2] + thisPathway.calcInputRczDimsToProduceOutputFmsOfCompatibleDims(thisPathWayKernelDimensions, dimsOfOutputFrom1stPathwayVal)
            inputToPathwayShapeTest = [self.batchSize["test"], numberOfImageChannelsPath2] + thisPathway.calcInputRczDimsToProduceOutputFmsOfCompatibleDims(thisPathWayKernelDimensions, dimsOfOutputFrom1stPathwayTest)
            
            thisPathway.makeLayersOfThisPathwayAndReturnDimensionsOfOutputFM(log,
                                                                     inputToPathwayTrain,
                                                                     inputToPathwayVal,
                                                                     inputToPathwayTest,
                                                                     inputToPathwayShapeTrain,
                                                                     inputToPathwayShapeVal,
                                                                     inputToPathwayShapeTest,
                                                                     thisPathWayNKerns,
                                                                     thisPathWayKernelDimensions,
                                                                     
                                                                     convWInitMethod,
                                                                     thisPathwayUseBnPerLayer,
                                                                     movingAvForBnOverXBatches,
                                                                     thisPathwayActivFuncPerLayer,

self.batchSize_seg,
self.batchSize_adv,
                                                                             dropoutRatesForAllPathways[thisPathwayType],
                                                                     
                                                                     maxPoolingParamsStructure[thisPathwayType],
                                                                     
                                                                     indicesOfLowerRankLayersPerPathway[thisPathwayType],
                                                                     ranksOfLowerRankLayersForEachPathway[thisPathwayType],
                                                                     indicesOfLayersToConnectResidualsInOutput[thisPathwayType]                                     )
            # Skip connections to end of pathway.
            thisPathway.makeMultiscaleConnectionsForLayerType(convLayersToConnectToFirstFcForMultiscaleFromAllLayerTypes[thisPathwayType])
            
            # this creates essentially the "upsampling layer"
            ##==========================Upsample the activations from subsampled pathway in the function==============================================##
            thisPathway.upsampleOutputToNormalRes(upsamplingScheme="repeat",
                                                  shapeToMatchInRczTrain=dimsOfOutputFrom1stPathwayTrain,
                                                  shapeToMatchInRczVal=dimsOfOutputFrom1stPathwayVal,
                                                  shapeToMatchInRczTest=dimsOfOutputFrom1stPathwayTest,
                                                  shapeToMatchForDDTrain=dimsOfOutputForDDFrom1stPathwayTrain, 
                                                  shapeToMatchForDDVal=dimsOfOutputForDDFrom1stPathwayVal,
                                                  shapeToMatchForDDTest=dimsOfOutputForDDFrom1stPathwayTest)
            # Get the outputDD from subsampled pathway.
            [UpsampledOutputForDDFromSubsampledPathwayTrain, 
            UpsampledOutputForDDFromSubsampledPathwayVal, 
            UpsampledOutputForDDFromSubsampledPathwayTest] = thisPathway.getOutputDD()

            [UpsampledOutputForDDFromSubsampledPathwayShapeTrain, 
            UpsampledOutputForDDFromSubsampledPathwayShapeVal, 
            UpsampledOutputForDDFromSubsampledPathwayShapeTest] = thisPathway.getShapeOfOutputDD()
            ##==========================================================================================##



        #====================================CONCATENATE the output of the 2 cnn-pathways=============================
        inputToFirstFcLayerTrain = None; inputToFirstFcLayerVal = None; inputToFirstFcLayerTest = None; numberOfFmsOfInputToFirstFcLayer = 0
        for path_i in range(len(self.pathways)) :
            [outputNormResOfPathTrain, outputNormResOfPathVal, outputNormResOfPathTest] = self.pathways[path_i].getOutputAtNormalRes()
            [dimsOfOutputNormResOfPathTrain, dimsOfOutputNormResOfPathVal, dimsOfOutputNormResOfPathTest] = self.pathways[path_i].getShapeOfOutputAtNormalRes()
            
            inputToFirstFcLayerTrain =  tf.concat([inputToFirstFcLayerTrain, outputNormResOfPathTrain], axis=1) if path_i != 0 else outputNormResOfPathTrain
            inputToFirstFcLayerVal = tf.concat([inputToFirstFcLayerVal, outputNormResOfPathVal], axis=1) if path_i != 0 else outputNormResOfPathVal
            inputToFirstFcLayerTest = tf.concat([inputToFirstFcLayerTest, outputNormResOfPathTest], axis=1) if path_i != 0 else outputNormResOfPathTest
            numberOfFmsOfInputToFirstFcLayer += dimsOfOutputNormResOfPathTrain[1]
            
        #======================= Make the Fully Connected Layers =======================
        thisPathway = FcPathway()
        self.pathways.append(thisPathway)
        thisPathwayType = thisPathway.pType()
        
        # This is the shape of the kernel in the first FC layer.
        # NOTE: If there is no hidden FC layer, this kernel is used in the Classification layer then.
        # Originally it was 1x1x1 only. The pathways themselves where taking care of the receptive field.
        # However I can now define it larger (eg 3x3x3), in case it helps combining the multiresolution features better/smoother.
        # The convolution is seamless, ie same shape output/input, by mirror padding the input.
        firstFcLayerAfterConcatenationKernelShape = self.kernelDimensionsFirstFcLayer
        voxelsToPadPerDim = [ kernelDim - 1 for kernelDim in firstFcLayerAfterConcatenationKernelShape ]
        log.print3("DEBUG: Shape of the kernel of the first FC layer is : " + str(firstFcLayerAfterConcatenationKernelShape))
        log.print3("DEBUG: Input to the FC Pathway will be padded by that many voxels per dimension: " + str(voxelsToPadPerDim))
        inputToPathwayTrain = padImageWithMirroring(inputToFirstFcLayerTrain, voxelsToPadPerDim)
        inputToPathwayVal = padImageWithMirroring(inputToFirstFcLayerVal, voxelsToPadPerDim)
        inputToPathwayTest = padImageWithMirroring(inputToFirstFcLayerTest, voxelsToPadPerDim)
        inputToPathwayShapeTrain = [self.batchSize["train"], numberOfFmsOfInputToFirstFcLayer] + dimsOfOutputFrom1stPathwayTrain[2:5]
        inputToPathwayShapeVal = [self.batchSize["val"], numberOfFmsOfInputToFirstFcLayer] + dimsOfOutputFrom1stPathwayVal[2:5]
        inputToPathwayShapeTest = [self.batchSize["test"], numberOfFmsOfInputToFirstFcLayer] + dimsOfOutputFrom1stPathwayTest[2:5]
        for rcz_i in range(3) : 
            inputToPathwayShapeTrain[2+rcz_i] += voxelsToPadPerDim[rcz_i]
            inputToPathwayShapeVal[2+rcz_i] += voxelsToPadPerDim[rcz_i]
            inputToPathwayShapeTest[2+rcz_i] += voxelsToPadPerDim[rcz_i]
        
        thisPathWayNKerns = fcLayersFMs + [self.num_classes]
        thisPathWayKernelDimensions = [firstFcLayerAfterConcatenationKernelShape] + [[1, 1, 1]] * (len(thisPathWayNKerns) - 1)
        
        thisPathwayNumOfLayers = len(thisPathWayNKerns)
        thisPathwayUseBnPerLayer = [movingAvForBnOverXBatches > 0] * thisPathwayNumOfLayers
        thisPathwayUseBnPerLayer[0] = applyBnToInputOfPathways[thisPathwayType] if movingAvForBnOverXBatches > 0 else False  # For the 1st layer, ask specific flag.
        
        thisPathwayActivFuncPerLayer = [activationFunc] * thisPathwayNumOfLayers
        thisPathwayActivFuncPerLayer[0] = "linear" if thisPathwayType != pt.FC else activationFunc  # To not apply activation on raw input. -1 is linear activation.
        
        thisPathway.makeLayersOfThisPathwayAndReturnDimensionsOfOutputFM(log,
                                                                         inputToPathwayTrain,
                                                                         inputToPathwayVal,
                                                                         inputToPathwayTest,
                                                                         inputToPathwayShapeTrain,
                                                                         inputToPathwayShapeVal,
                                                                         inputToPathwayShapeTest,
                                                                         
                                                                         thisPathWayNKerns,
                                                                         thisPathWayKernelDimensions,
                                                                         
                                                                         convWInitMethod,
                                                                         thisPathwayUseBnPerLayer,
                                                                         movingAvForBnOverXBatches,
                                                                         thisPathwayActivFuncPerLayer,
                                                                         
self.batchSize_seg,
                                                                         self.batchSize_adv,
                                                                         dropoutRatesForAllPathways[thisPathwayType],
                                                                         
                                                                         maxPoolingParamsStructure[thisPathwayType],
                                                                         
                                                                         indicesOfLowerRankLayersPerPathway[thisPathwayType],
                                                                         ranksOfLowerRankLayersForEachPathway[thisPathwayType],
                                                                         indicesOfLayersToConnectResidualsInOutput[thisPathwayType])

        ##=================================Get the DDoutput of FC Pathway======================================================##
        [DDOutputFromFcPathwayTrain, DDOutputFromFcPathwayVal, DDOutputFromFcPathwayTest] = thisPathway.getOutputDD()
        [DDOutputFromFcPathwayShapeTrain, DDOutputFromFcPathwayShapeVal, DDOutputFromFcPathwayShapeTest] = thisPathway.getShapeOfOutputDD()
        ##=====================================================================================================================##
        # =========== Make the final Target Layer (softmax, regression, whatever) ==========
        log.print3("Adding the final Softmax Target layer...")
        
        self.finalTargetLayer = self._getClassificationLayer()
        self.finalTargetLayer.makeLayer(rng, self.getFcPathway().getLayer(-1), self.batchSize_seg, softmaxTemperature)
        (self._output_gt_tensor_feeds['train']['y_gt'],
         self._output_gt_tensor_feeds['val']['y_gt']) = self.finalTargetLayer.get_output_gt_tensor_feed()
        
        #log.print3("Finished building the CNN's model.")


##=============================================================================================================##


        #====================== Make the Domain Discriminator ========================#
        DDnkerns = [100, 100, 100, 100, 2] #
        # what is the number of kernels in the last classification layer. 1 or 2??????????????
        DDkernelDimensions = [[3,3,3], [3,3,3], [3,3,1], [3,3,1], [1,1,1]] # a list
        dropoutRatesDDPathway = [0.0, 0.0, 0.0, 0.0, 0.5] # the last layer serves as a fully connected layer
                                                          # as well as a classification layer.
                                                          #not sure what it should be. 
        maxPoolingParamsDD = [[], [], [], [], []]
        thisPathway = DDPathway()
        self.pathways.append(thisPathway)
        thisPathwayType = thisPathway.pType()
        
        #========================Get the input For Domain Discriminator================================#
        middleResultTrain = tf.concat([OutputForDDFrom1stPathwayTrain, UpsampledOutputForDDFromSubsampledPathwayTrain], axis=1)
        #Fc Activations, need to Crop to march it!
        partTrain = getMiddlePartOfFms(middleResultTrain, DDOutputFromFcPathwayShapeTrain[2:])
        inputToPathwayTrain = tf.concat([DDOutputFromFcPathwayTrain, partTrain], axis=1)

        middleResultVal = tf.concat([OutputForDDFrom1stPathwayVal, UpsampledOutputForDDFromSubsampledPathwayVal], axis=1)
        partVal = getMiddlePartOfFms(middleResultVal, DDOutputFromFcPathwayShapeVal[2:])
        inputToPathwayVal = tf.concat([DDOutputFromFcPathwayVal, partVal], axis=1)

        middleResultTest = tf.concat([OutputForDDFrom1stPathwayTest, UpsampledOutputForDDFromSubsampledPathwayTest], axis=1)
        partTest = getMiddlePartOfFms(middleResultTest, DDOutputFromFcPathwayShapeTest[2:])
        inputToPathwayTest = tf.concat([DDOutputFromFcPathwayTest, partTest], axis=1)

        imagePartDimensionsTraining = DDOutputFromFcPathwayShapeTrain[2:]
        imagePartDimensionsValidation = DDOutputFromFcPathwayShapeVal[2:]
        imagePartDimensionsTesting = DDOutputFromFcPathwayShapeTest[2:]

        numOfFms = UpsampledOutputForDDFromSubsampledPathwayShapeTrain[1] + dimsOfOutputForDDFrom1stPathwayTrain[1] + DDOutputFromFcPathwayShapeTrain[1]
        
        log.print3("DDPathway final Input dimensions Check     !!!"+str(numOfFms))
                   
        ##================================================================#
        ##DDbatchSize, imagePartDimensionsTrain is the size of image segment.
        
        inputToPathwayShapeTrain = [self.batchSize_adv["train"]*2] + [numOfFms] + imagePartDimensionsTraining
        inputToPathwayShapeVal = [self.batchSize_adv["val"]*2] + [numOfFms] + imagePartDimensionsValidation
        inputToPathwayShapeTest = [self.batchSize_adv["test"]*2] + [numOfFms] + imagePartDimensionsTesting
        
        thisPathWayNKerns = DDnkerns
        thisPathWayKernelDimensions = DDkernelDimensions
        
        thisPathwayNumOfLayers = len(thisPathWayNKerns)

        ##this movingAvForBnOverXBatches comes from a param rollAverageForBNOverThatManyBatches in config file.
        ##We will temporarily use the same one as the segmenter
        thisPathwayUseBnPerLayer = [movingAvForBnOverXBatches > 0] * thisPathwayNumOfLayers
        thisPathwayUseBnPerLayer[0] = True if movingAvForBnOverXBatches > 0 else False  # For the 1st layer, ask specific flag.
        
        thisPathwayActivFuncPerLayer = [activationFunc] * thisPathwayNumOfLayers
        thisPathwayActivFuncPerLayer[0] = "linear" if thisPathwayType != pt.FC else activationFunc  # To not apply activation on raw input. -1 is linear activation.



        thisPathway.makeLayersOfDDPathwayAndReturnDimensionsOfOutputFM(log,
                                                                         inputToPathwayTrain,
                                                                         inputToPathwayVal,
                                                                         inputToPathwayTest,
                                                                         inputToPathwayShapeTrain,
                                                                         inputToPathwayShapeVal,
                                                                         inputToPathwayShapeTest,
                                                                         
                                                                         thisPathWayNKerns,
                                                                         thisPathWayKernelDimensions,
                                                                         
                                                                         convWInitMethod,
                                                                         thisPathwayUseBnPerLayer,
                                                                         movingAvForBnOverXBatches,
                                                                         thisPathwayActivFuncPerLayer,

                                                                         dropoutRatesDDPathway,#!!!!!!!!!!
                                                                         
                                                                         maxPoolingParamsDD,#!!!!!!!!!!!
                                                                         )
       
        [dimsOfOutputFromthisPathwayTrain, dimsOfOutputFromthisPathwayVal, dimsOfOutputFromthisPathwayTest] = thisPathway.getShapeOfOutput()
        log.print3("Output Shape Of DD Pathway Train "+str(dimsOfOutputFromthisPathwayTrain)+" Val "+str(dimsOfOutputFromthisPathwayVal) + " Test "+str(dimsOfOutputFromthisPathwayTest))
        
       ##=======================================================================================================## 
      ## =========== Make the final Target Layer (softmax) to classify which 
        ###domain this sample belongs to================#
        
        log.print3("Adding the final Softmax Target layer...")
        
        self.DDfinalTargetLayer = self._getClassificationLayer()
        self.DDfinalTargetLayer.makeDDLayer(rng, self.getDDPathway().getLayer(-1), softmaxTemperature)
        
        #Ground truth labels
        #Placeholders!
        (self._DDoutput_gt_tensor_feeds['train']['y_gt'],
         self._DDoutput_gt_tensor_feeds['val']['y_gt']) = self.DDfinalTargetLayer.get_DDoutput_gt_tensor_feed()
        
        log.print3("Finished building the CNN's model.")
    