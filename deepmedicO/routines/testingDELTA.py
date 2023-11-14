# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division

import time
import numpy as np
import math
import tensorflow as tf

from deepmedicO.logging.accuracyMonitor import AccuracyOfEpochMonitorSegmentation
from deepmedicO.dataManagement.sampling import load_imgs_of_single_case
from deepmedicO.dataManagement.sampling import getCoordsOfAllSegmentsOfAnImage
from deepmedicO.dataManagement.sampling import extractDataOfSegmentsUsingSampledSliceCoords
from deepmedicO.image.io import savePredImgToNiiWithOriginalHdr, saveFmImgToNiiWithOriginalHdr, save4DImgWithAllFmsToNiiWithOriginalHdr
from deepmedicO.image.processing import unpadCnnOutputs

from deepmedicO.neuralnet.pathwayTypes import PathwayTypes as pt
from deepmedicO.logging.utils import strListFl4fNA, getMeanPerColOf2dListExclNA


# Main routine for testing.
def performInferenceOnWholeVolumes(
                            sessionTf,
                            cnn3d,
                            log,
                            val_or_test,
                            savePredictionImagesSegmentationAndProbMapsList,

                            listOfFilepathsToEachChannelOfEachPatient,
                            
                            providedGtLabelsBool, #boolean. DSC calculation will be performed if this is provided.
                            listOfFilepathsToGtLabelsOfEachPatient,
                            
                            providedRoiMaskForFastInfBool,
                            listOfFilepathsToRoiMaskFastInfOfEachPatient,
                            
                            listOfNamesToGiveToPredictionsIfSavingResults,
                            
                            #----Preprocessing------
                            padInputImagesBool,
                            
                            useSameSubChannelsAsSingleScale,
                            listOfFilepathsToEachSubsampledChannelOfEachPatient,
                            
                            #--------For FM visualisation---------
                            saveIndividualFmImagesForVisualisation,
                            saveMultidimensionalImageWithAllFms,
                            indicesOfFmsToVisualisePerPathwayTypeAndPerLayer,#NOTE: saveIndividualFmImagesForVisualisation should contain an entry per pathwayType, even if just []. If not [], the list should contain one entry per layer of the pathway, even if just []. The layer entries, if not [], they should have to integers, lower and upper FM to visualise. Excluding the highest index.
                            listOfNamesToGiveToFmVisualisationsIfSaving,
                            saver_net,
                            chkpt_fname
                            ) :
    validation_or_testing_str = "Validation" if val_or_test == "val" else "Testing"
    log.print3("###########################################################################################################")
    log.print3("############################# Starting full Segmentation of " + str(validation_or_testing_str) + " subjects ##########################")
    log.print3("###########################################################################################################")
    
    start_time = time.time()
    
    NA_PATTERN = AccuracyOfEpochMonitorSegmentation.NA_PATTERN
    
    NUMBER_OF_CLASSES = cnn3d.num_classes
    
    total_number_of_images = len(listOfFilepathsToEachChannelOfEachPatient)    
    batch_size = cnn3d.batchSize["test"]
    
    #one dice score for whole + for each class)
    # A list of dimensions: total_number_of_images X NUMBER_OF_CLASSES
    diceCoeffs1 = [ [-1] * NUMBER_OF_CLASSES for i in range(total_number_of_images) ] #AllpredictedLes/AllLesions
    diceCoeffs2 = [ [-1] * NUMBER_OF_CLASSES for i in range(total_number_of_images) ] #predictedInsideRoiMask/AllLesions
    diceCoeffs3 = [ [-1] * NUMBER_OF_CLASSES for i in range(total_number_of_images) ] #predictedInsideRoiMask/ LesionsInsideRoiMask (for comparisons)
    
    recFieldCnn = cnn3d.recFieldCnn
    
    #stride is how much I move in each dimension to acquire the next imagePart. 
    #I move exactly the number I segment in the centre of each image part (originally this was 9^3 segmented per imagePart).
    numberOfCentralVoxelsClassified = cnn3d.finalTargetLayer.outputShape["test"][2:]
    strideOfImagePartsPerDimensionInVoxels = numberOfCentralVoxelsClassified
    
    rczHalfRecFieldCnn = [ (recFieldCnn[i]-1)//2 for i in range(3) ]
    
    #Find the total number of feature maps that will be created:
    #NOTE: saveIndividualFmImagesForVisualisation should contain an entry per pathwayType, even if just []. If not [], the list should contain one entry per layer of the pathway, even if just []. The layer entries, if not [], they should have to integers, lower and upper FM to visualise.
    if saveIndividualFmImagesForVisualisation or saveMultidimensionalImageWithAllFms:
        totalNumberOfFMsToProcess = 0
        for pathway in cnn3d.pathways :
            indicesOfFmsToVisualisePerLayerOfCertainPathway = indicesOfFmsToVisualisePerPathwayTypeAndPerLayer[ pathway.pType() ]
            if indicesOfFmsToVisualisePerLayerOfCertainPathway != [] :
                for layer_i in range(len(pathway.getLayers())) :
                    indicesOfFmsToVisualiseForCertainLayerOfCertainPathway = indicesOfFmsToVisualisePerLayerOfCertainPathway[layer_i]
                    if indicesOfFmsToVisualiseForCertainLayerOfCertainPathway!=[] :
                        #If the user specifies to grab more feature maps than exist (eg 9999), correct it, replacing it with the number of FMs in the layer.
                        numberOfFeatureMapsInThisLayer = pathway.getLayer(layer_i).getNumberOfFeatureMaps()
                        indicesOfFmsToVisualiseForCertainLayerOfCertainPathway[1] = min(indicesOfFmsToVisualiseForCertainLayerOfCertainPathway[1], numberOfFeatureMapsInThisLayer)
                        totalNumberOfFMsToProcess += indicesOfFmsToVisualiseForCertainLayerOfCertainPathway[1] - indicesOfFmsToVisualiseForCertainLayerOfCertainPathway[0]
    

    ##================================ADDED====================================##
    channel_weights = []
    pathway = cnn3d.getFcPathway()
    topLayers = [1]
    num_of_layers = len(topLayers)
    for i in range(num_of_layers):
        channel = pathway.getLayer(topLayers[i]).outputShape["train"][1]
        channel_weights.append([0] * channel)

    total_number_of_images = 4
    ##================================END========================================##

    for image_i in range(total_number_of_images) :
        log.print3("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        log.print3("~~~~~~~~~~~~~~~~~~~~ Segmenting subject with index #"+str(image_i)+" ~~~~~~~~~~~~~~~~~~~~")
        
        #load the image channels in cpu
        
        [imageChannels, #a nparray(channels,dim0,dim1,dim2)
        gtLabelsImage, #only for accurate/correct DICE1-2 calculation
        roiMask,
        arrayWithWeightMapsWhereToSampleForEachCategory, #only used in training. Placeholder here.
        allSubsampledChannelsOfPatientInNpArray,  #a nparray(channels,dim0,dim1,dim2)
        tupleOfPaddingPerAxesLeftRight #( (padLeftR, padRightR), (padLeftC,padRightC), (padLeftZ,padRightZ)). All 0s when no padding.
        ] = load_imgs_of_single_case(
                                    log,
                                    "test",
                                    
                                    image_i,
                                    
                                    listOfFilepathsToEachChannelOfEachPatient,
                                    
                                    providedGtLabelsBool,
                                    listOfFilepathsToGtLabelsOfEachPatient,
                                    num_classes = cnn3d.num_classes,
                                    
                                    providedWeightMapsToSampleForEachCategory = False, # Says if weightMaps are provided. If true, must provide all. Placeholder in testing.
                                    forEachSamplingCategory_aListOfFilepathsToWeightMapsOfEachPatient = "placeholder", # Placeholder in testing.
                                    
                                    providedRoiMaskBool = providedRoiMaskForFastInfBool,
                                    listOfFilepathsToRoiMaskOfEachPatient = listOfFilepathsToRoiMaskFastInfOfEachPatient,
                                    
                                    useSameSubChannelsAsSingleScale = useSameSubChannelsAsSingleScale,
                                    usingSubsampledPathways = cnn3d.numSubsPaths > 0,
                                    listOfFilepathsToEachSubsampledChannelOfEachPatient = listOfFilepathsToEachSubsampledChannelOfEachPatient,
                                    
                                    padInputImagesBool = padInputImagesBool,
                                    cnnReceptiveField = recFieldCnn, # only used if padInputsBool
                                    dimsOfPrimeSegmentRcz = cnn3d.pathways[0].getShapeOfInput("test")[2:], # only used if padInputsBool
                                    
                                    reflectImageWithHalfProb = [0,0,0]
                                    )
        niiDimensions = list(imageChannels[0].shape)
        #The predicted probability-maps for the whole volume, one per class. Will be constructed by stitching together the predictions from each segment.
        predProbMapsPerClass = np.zeros([NUMBER_OF_CLASSES]+niiDimensions, dtype = "float32")
        #create the big array that will hold all the fms (for feature extraction, to save as a big multi-dim image).
        if saveIndividualFmImagesForVisualisation or saveMultidimensionalImageWithAllFms:
            multidimensionalImageWithAllToBeVisualisedFmsArray =  np.zeros([totalNumberOfFMsToProcess] + niiDimensions, dtype = "float32")
            
        # Tile the image and get all slices of the segments that it fully breaks down to.
        [sliceCoordsOfSegmentsInImage] = getCoordsOfAllSegmentsOfAnImage(log=log,
                                                                        dimsOfPrimarySegment=cnn3d.pathways[0].getShapeOfInput("test")[2:],
                                                                        strideOfSegmentsPerDimInVoxels=strideOfImagePartsPerDimensionInVoxels,
                                                                        batch_size = batch_size,
                                                                        channelsOfImageNpArray = imageChannels,#chans,niiDims
                                                                        roiMask = roiMask
                                                                        )
        log.print3("Starting to segment each image-part by calling the cnn.cnnTestModel(i). This part takes a few mins per volume...")
        
        
        totalNumberOfImagePartsToProcessForThisImage = len(sliceCoordsOfSegmentsInImage)
        log.print3("Total number of Segments to process:"+str(totalNumberOfImagePartsToProcessForThisImage))
        
        imagePartOfConstructedProbMap_i = 0
        imagePartOfConstructedFeatureMaps_i = 0
        number_of_batches = totalNumberOfImagePartsToProcessForThisImage//batch_size
        extractTimePerSubject = 0; loadingTimePerSubject = 0; fwdPassTimePerSubject = 0
        for batch_i in range(number_of_batches) : #batch_size = how many image parts in one batch. Has to be the same with the batch_size it was created with. This is no problem for testing. Could do all at once, or just 1 image part at time.
            
            printProgressStep = max(1, number_of_batches//5)
            if batch_i%printProgressStep == 0:
                log.print3("Processed "+str(batch_i*batch_size)+"/"+str(number_of_batches*batch_size)+" Segments.")
                
            # Extract the data for the segments of this batch. ( I could modularize extractDataOfASegmentFromImagesUsingSampledSliceCoords() of training and use it here as well. )
            start_extract_time = time.time()
            sliceCoordsOfSegmentsInBatch = sliceCoordsOfSegmentsInImage[ batch_i*batch_size : (batch_i+1)*batch_size ]
            [channsOfSegmentsPerPath] = extractDataOfSegmentsUsingSampledSliceCoords(cnn3d=cnn3d,
                                                                                    sliceCoordsOfSegmentsToExtract=sliceCoordsOfSegmentsInBatch,
                                                                                    channelsOfImageNpArray=imageChannels,#chans,niiDims
                                                                                    channelsOfSubsampledImageNpArray=allSubsampledChannelsOfPatientInNpArray,
                                                                                    recFieldCnn=recFieldCnn
                                                                                    )
            end_extract_time = time.time()
            extractTimePerSubject += end_extract_time - start_extract_time
            
            # ======= Run the inference ============
            
            
            # No loading of data in bulk as in training, cause here it's only 1 batch per iteration.
            start_loading_time = time.time()
            feeds = cnn3d.get_main_feeds('test')
            feeds_dict = { feeds['x'] : np.asarray(channsOfSegmentsPerPath[0], dtype='float32') }
            for path_i in range(len(channsOfSegmentsPerPath[1:])) :
                feeds_dict.update( { feeds['x_sub_'+str(path_i)]: np.asarray(channsOfSegmentsPerPath[1+path_i], dtype='float32') } )
            end_loading_time = time.time()
            loadingTimePerSubject += end_loading_time - start_loading_time
            
            start_testing_time = time.time()
            # Forward pass
            
##==================================================================================================================================================##
            ops_to_fetch = cnn3d.get_main_ops('test')
            list_of_ops = [ ops_to_fetch['pred_probs'] ] + ops_to_fetch['list_of_fms_per_layer']
            
            # featureMapsOfEachLayerAndPredictionProbabilitiesAtEndForATestBatch = cnn3d.cnnTestAndVisualiseAllFmsFunction( *input_args_to_net )
            featureMapsOfEachLayerAndPredictionProbabilitiesAtEndForATestBatch = sessionTf.run( fetches=list_of_ops, feed_dict=feeds_dict )
            prediction_original = featureMapsOfEachLayerAndPredictionProbabilitiesAtEndForATestBatch[0]
            
            print(len(channel_weights[0]))
            
            if batch_i < number_of_batches-1:
                continue
            
            for i in range(num_of_layers):
                for j in range(len(channel_weights[i])):
                    
                    #W = pathway.getLayer(topLayers[i]).getTrainableParams()[0]
                    #Wshape = tf.shape(W)
                    saver_net.restore(sessionTf, chkpt_fname)
                    collection_vars_net = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="net")
                    for v in collection_vars_net:
                        if v.name == 'net/W_25:0':
                            W = v
                            break
                    Wshape = tf.shape(W)
                    zeros = tf.zeros([Wshape[1],Wshape[2], Wshape[3], Wshape[4]], tf.float32)
                    opAssign1 = tf.assign(W[j, :, :, :, :], zeros) 
                    sessionTf.run(opAssign1)
                    
                    print(str(W))
                    
                    featureMapsOfEachLayerAndPredictionProbabilitiesAtEndForATestBatch = sessionTf.run( fetches=list_of_ops, feed_dict=feeds_dict )
                    predictionForATestBatch = featureMapsOfEachLayerAndPredictionProbabilitiesAtEndForATestBatch[0]
                    imagePartOfConstructedProbMap_i = 0
                    
                    #~~~~~~~~~~~~~~~~CONSTRUCT THE PREDICTED PROBABILITY MAPS~~~~~~~~~~~~~~
                    #From the results of this batch, create the prediction image by putting the predictions to the correct place in the image.
                    for imagePart_in_this_batch_i in range(batch_size) :
                        #Now put the label-cube in the new-label-segmentation-image, at the correct position. 
                        #The very first label goes not in index 0,0,0 but half-patch further away! At the position of the central voxel of the top-left patch!
                        sliceCoordsOfThisSegment = sliceCoordsOfSegmentsInImage[imagePartOfConstructedProbMap_i]
                        coordsOfTopLeftVoxelForThisPart = [ sliceCoordsOfThisSegment[0][0], sliceCoordsOfThisSegment[1][0], sliceCoordsOfThisSegment[2][0] ]
                        predProbMapsPerClass[
                                :,
                                coordsOfTopLeftVoxelForThisPart[0] + rczHalfRecFieldCnn[0] : coordsOfTopLeftVoxelForThisPart[0] + rczHalfRecFieldCnn[0] + strideOfImagePartsPerDimensionInVoxels[0],
                                coordsOfTopLeftVoxelForThisPart[1] + rczHalfRecFieldCnn[1] : coordsOfTopLeftVoxelForThisPart[1] + rczHalfRecFieldCnn[1] + strideOfImagePartsPerDimensionInVoxels[1],
                                coordsOfTopLeftVoxelForThisPart[2] + rczHalfRecFieldCnn[2] : coordsOfTopLeftVoxelForThisPart[2] + rczHalfRecFieldCnn[2] + strideOfImagePartsPerDimensionInVoxels[2],
                                ] = predictionForATestBatch[imagePart_in_this_batch_i]
                        predProbMapsPerClassO = np.zeros([NUMBER_OF_CLASSES]+niiDimensions, dtype = "float32")
                        predProbMapsPerClassO[
                                :,
                                coordsOfTopLeftVoxelForThisPart[0] + rczHalfRecFieldCnn[0] : coordsOfTopLeftVoxelForThisPart[0] + rczHalfRecFieldCnn[0] + strideOfImagePartsPerDimensionInVoxels[0],
                                coordsOfTopLeftVoxelForThisPart[1] + rczHalfRecFieldCnn[1] : coordsOfTopLeftVoxelForThisPart[1] + rczHalfRecFieldCnn[1] + strideOfImagePartsPerDimensionInVoxels[1],
                                coordsOfTopLeftVoxelForThisPart[2] + rczHalfRecFieldCnn[2] : coordsOfTopLeftVoxelForThisPart[2] + rczHalfRecFieldCnn[2] + strideOfImagePartsPerDimensionInVoxels[2],
                                ] = prediction_original[imagePart_in_this_batch_i]
                        imagePartOfConstructedProbMap_i += 1
                    #~~~~~~~~~~~~~FINISHED CONSTRUCTING THE PREDICTED PROBABILITY MAPS~~~~~~~
                    
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Evaluate DSC~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    predSegmentation = np.argmax(predProbMapsPerClass, axis=0) #The segmentation.
                    unpaddedPredSegmentation = predSegmentation if not padInputImagesBool else unpadCnnOutputs(predSegmentation, tupleOfPaddingPerAxesLeftRight)
                    predSegmentationO = np.argmax(predProbMapsPerClassO, axis=0) #The segmentation.
                    unpaddedPredSegmentationO = predSegmentationO if not padInputImagesBool else unpadCnnOutputs(predSegmentationO, tupleOfPaddingPerAxesLeftRight)
                    
                    #log.print3("+++++++++++++++++++++ Reporting Segmentation Metrics for the subject #" + str(image_i) + "++++++++++++++++++++++++++")
                    #Unpad whatever needed.
                    unpaddedGtLabelsImage = gtLabelsImage if not padInputImagesBool else unpadCnnOutputs(gtLabelsImage, tupleOfPaddingPerAxesLeftRight)
                    #calculate DSC per class.
                    for class_i in range(0, NUMBER_OF_CLASSES) :
                        if class_i == 0 : #in this case, do the evaluation for the segmentation of the WHOLE FOREGROUND (ie, all classes merged except background)
                            binaryPredSegmClassI = unpaddedPredSegmentation > 0 # Merge every class except the background (assumed to be label == 0 )
                            binaryPredSegmClassIO = unpaddedPredSegmentationO > 0
                            binaryGtLabelClassI = unpaddedGtLabelsImage > 0
                        else :
                            binaryPredSegmClassI = unpaddedPredSegmentation == class_i
                            binaryPredSegmClassIO = unpaddedPredSegmentationO == class_i
                            binaryGtLabelClassI = unpaddedGtLabelsImage == class_i

                        #Calculate the 3 Dices. Dice1 = Allpredicted/allLesions, Dice2 = PredictedWithinRoiMask / AllLesions , Dice3 = PredictedWithinRoiMask / LesionsInsideRoiMask.
                        #Dice1 = Allpredicted/allLesions
                        diceCoeff1 = calculateDiceCoefficient(binaryPredSegmClassI, binaryGtLabelClassI)
                        diceCoeffs1[image_i][class_i] = diceCoeff1 if diceCoeff1 != -1 else NA_PATTERN
                        
                        diceCoeff1O = calculateDiceCoefficient(binaryPredSegmClassIO, binaryGtLabelClassI)
                        

                    log.print3("ACCURACY: (" + str(validation_or_testing_str) + ") The Per-Class DICE Coefficients for subject with index #"+str(image_i)+" equal: DICE1="+strListFl4fNA(diceCoeffs1[image_i],NA_PATTERN))
                    #printExplanationsAboutDice(log)'''
            ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`##
                    
                    diff = np.mean((predProbMapsPerClassO[0] - predProbMapsPerClass[0]) ** 2)
                    hist = channel_weights[i][j]
                    channel_weights[i][j] = 1.0 * (image_i * hist + diff) / (image_i + 1)
                    print(str(channel_weights) + " " + str(hist))
                    
                    #print("sec")
                    #channel_weights[i][j] = diff
                    #print('%s:%d %.4f %.4f' % (name, j, diff, filter_weight[layer_id][j]))
                    #opAssign2 = tf.assign(W, tmp)  
                    #sessionTf.run(opAssign2)
                    
                    #collection_vars_net = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="net")
                    #tf.variables_initializer(collection_vars_net[-2:]).run()
                    #saver_net = tf.train.Saver( var_list=collection_vars_net[:-2] )
                    
                    #log.print3("Cost: " + str(diff.eval(session=sessionTf)))
                    
            end_testing_time = time.time()
            fwdPassTimePerSubject += end_testing_time - start_testing_time


            # Then save these channel weights in a file
            for i in range(len(channel_weights[0])):
                with open("weights.txt", 'a+') as f:
                    f.writelines(str(channel_weights[0][i]))


            ##======================================================================================================##

        #================= EVALUATE DSC FOR EACH SUBJECT ========================
        '''if providedGtLabelsBool : # Ground Truth was provided for calculation of DSC. Do DSC calculation.
            log.print3("+++++++++++++++++++++ Reporting Segmentation Metrics for the subject #" + str(image_i) + " ++++++++++++++++++++++++++")
            #Unpad whatever needed.
            unpaddedGtLabelsImage = gtLabelsImage if not padInputImagesBool else unpadCnnOutputs(gtLabelsImage, tupleOfPaddingPerAxesLeftRight)
            #calculate DSC per class.
            for class_i in range(0, NUMBER_OF_CLASSES) :
                if class_i == 0 : #in this case, do the evaluation for the segmentation of the WHOLE FOREGROUND (ie, all classes merged except background)
                    binaryPredSegmClassI = unpaddedPredSegmentation > 0 # Merge every class except the background (assumed to be label == 0 )
                    binaryGtLabelClassI = unpaddedGtLabelsImage > 0
                else :
                    binaryPredSegmClassI = unpaddedPredSegmentation == class_i
                    binaryGtLabelClassI = unpaddedGtLabelsImage == class_i
                    
                binaryPredSegmClassIWithinRoi = binaryPredSegmClassI * unpaddedRoiMaskIfGivenElse1
                
                #Calculate the 3 Dices. Dice1 = Allpredicted/allLesions, Dice2 = PredictedWithinRoiMask / AllLesions , Dice3 = PredictedWithinRoiMask / LesionsInsideRoiMask.
                #Dice1 = Allpredicted/allLesions
                diceCoeff1 = calculateDiceCoefficient(binaryPredSegmClassI, binaryGtLabelClassI)
                diceCoeffs1[image_i][class_i] = diceCoeff1 if diceCoeff1 != -1 else NA_PATTERN
                #Dice2 = PredictedWithinRoiMask / AllLesions
                diceCoeff2 = calculateDiceCoefficient(binaryPredSegmClassIWithinRoi, binaryGtLabelClassI)
                diceCoeffs2[image_i][class_i] = diceCoeff2 if diceCoeff2 != -1 else NA_PATTERN
                #Dice3 = PredictedWithinRoiMask / LesionsInsideRoiMask
                diceCoeff3 = calculateDiceCoefficient(binaryPredSegmClassIWithinRoi, binaryGtLabelClassI * unpaddedRoiMaskIfGivenElse1)
                diceCoeffs3[image_i][class_i] = diceCoeff3 if diceCoeff3 != -1 else NA_PATTERN
                
            log.print3("ACCURACY: (" + str(validation_or_testing_str) + ") The Per-Class DICE Coefficients for subject with index #"+str(image_i)+" equal: DICE1="+strListFl4fNA(diceCoeffs1[image_i],NA_PATTERN)+" DICE2="+strListFl4fNA(diceCoeffs2[image_i],NA_PATTERN)+" DICE3="+strListFl4fNA(diceCoeffs3[image_i],NA_PATTERN))
            printExplanationsAboutDice(log)
            
    #================= Loops for all patients have finished. Now lets just report the average DSC over all the processed patients. ====================
    if providedGtLabelsBool and total_number_of_images>0 : # Ground Truth was provided for calculation of DSC. Do DSC calculation.
        log.print3("+++++++++++++++++++++++++++++++ Segmentation of all subjects finished +++++++++++++++++++++++++++++++++++")
        log.print3("+++++++++++++++++++++ Reporting Average Segmentation Metrics over all subjects ++++++++++++++++++++++++++")
        meanDiceCoeffs1 = getMeanPerColOf2dListExclNA(diceCoeffs1, NA_PATTERN)
        meanDiceCoeffs2 = getMeanPerColOf2dListExclNA(diceCoeffs2, NA_PATTERN)
        meanDiceCoeffs3 = getMeanPerColOf2dListExclNA(diceCoeffs3, NA_PATTERN)
        log.print3("ACCURACY: (" + str(validation_or_testing_str) + ") The Per-Class average DICE Coefficients over all subjects are: DICE1=" + strListFl4fNA(meanDiceCoeffs1, NA_PATTERN) + " DICE2="+strListFl4fNA(meanDiceCoeffs2, NA_PATTERN)+" DICE3="+strListFl4fNA(meanDiceCoeffs3, NA_PATTERN))
        printExplanationsAboutDice(log)
        
    end_time = time.time()
    log.print3("TIMING: "+validation_or_testing_str+" process took time: "+str(end_time-start_time)+"(s)")
    
    log.print3("###########################################################################################################")
    log.print3("############################# Finished full Segmentation of " + str(validation_or_testing_str) + " subjects ##########################")
    log.print3("###########################################################################################################")
'''

def calculateDiceCoefficient(predictedBinaryLabels, groundTruthBinaryLabels) :
    unionCorrectlyPredicted = predictedBinaryLabels * groundTruthBinaryLabels
    numberOfTruePositives = np.sum(unionCorrectlyPredicted)
    numberOfGtPositives = np.sum(groundTruthBinaryLabels)
    diceCoeff = (2.0 * numberOfTruePositives) / (np.sum(predictedBinaryLabels) + numberOfGtPositives) if numberOfGtPositives!=0 else -1
    return diceCoeff

def printExplanationsAboutDice(log) :
    log.print3("EXPLANATION: DICE1/2/3 are lists with the DICE per class. For Class-0, we calculate DICE for whole foreground, i.e all labels merged, except the background label=0. Useful for multi-class problems.")
    log.print3("EXPLANATION: DICE1 is calculated as segmentation over whole volume VS whole Ground Truth (GT). DICE2 is the segmentation within the ROI vs GT. DICE3 is segmentation within the ROI vs the GT within the ROI.")
    log.print3("EXPLANATION: If an ROI mask has been provided, you should be consulting DICE2 or DICE3.")

    