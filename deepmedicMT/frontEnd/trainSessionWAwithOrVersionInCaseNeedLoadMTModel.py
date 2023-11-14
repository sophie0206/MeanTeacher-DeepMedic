# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.


##For Mean teacher method WA_avg

from __future__ import absolute_import, print_function, division
import os

from deepmedicMT.frontEnd.session import Session
from deepmedicMT.frontEnd.configParsing.utils import getAbsPathEvenIfRelativeIsGiven
from deepmedicMT.frontEnd.configParsing.trainSessionParams import TrainSessionParameters
from deepmedicMT.frontEnd.sessHelpers import makeFoldersNeededForTrainingSession, handle_exception_tf_restore

from deepmedicMT.logging.utils import datetimeNowAsStr
from deepmedicMT.neuralnet.cnn3dWA import Cnn3d
from deepmedicMT.neuralnet.trainer import Trainer

from deepmedicMT.routines.training import do_training

import tensorflow as tf

class TrainSession(Session):
    
    def __init__(self, cfg):
        self._out_folder_models = None
        self._out_folder_preds = None
        self._out_folder_fms = None
        self._params = None # Compiled from cfg. Required for run()
        Session.__init__(self, cfg)
        
    def _make_sess_name(self):
        sess_name = TrainSessionParameters.getSessionName(  self._cfg[self._cfg.SESSION_NAME] )
        return sess_name
    
    def make_output_folders(self):
        [self._log_folder_abs,
         self._out_folder_models,
         self._out_folder_preds,
         self._out_folder_fms] = makeFoldersNeededForTrainingSession( self._main_out_folder_abs, self._sess_name )
    
    
    def _print_vars_in_collection(self, collection, coll_name="no_name"):
        self._log.print3("")
        self._log.print3("==== Printing variables of collection [" +str(coll_name) + "] ====")
        for entry in collection:
            self._log.print3(str(entry))
        self._log.print3("==== Done printing variables of collection. ====\n")
    
    
    def compile_session_params_from_cfg(self, *args):
        (model_params,) = args
        
        self._params = TrainSessionParameters(
                                    log = self._log,
                                    mainOutputAbsFolder = self._main_out_folder_abs,
                                    folderForSessionCnnModels = self._out_folder_models,
                                    folderForPredictionsVal = self._out_folder_preds,
                                    folderForFeaturesVal = self._out_folder_fms,
                                    num_classes = model_params.numberClasses,
                                    model_name = model_params.cnnModelName,
                                    cfg = self._cfg )
        
        self._log.print3("")
        self._log.print3("=============   NEW TRAINING SESSION     ==============\n")
        self._params.print_params()
        self._log.print3("=======================================================\n")
        
        return self._params
    
    
    def run_session(self, *args):
        (sess_device,
         model_params,
         reset_trainer) = args
        
        graphTf = tf.Graph()
        
        with graphTf.as_default():
            with graphTf.device(sess_device): # Explicit device assignment, throws an error if GPU is specified but not available.
                self._log.print3("=========== Making the CNN graph... ===============")
                cnn3d = Cnn3d()
                cnn3dT = Cnn3d()
                with tf.variable_scope("net"): 
                    cnn3d.make_cnn_model( *model_params.get_args_for_arch() )
                    # I have now created the CNN graph. But not yet the Optimizer's graph.
                with tf.variable_scope("tch_net"):
                    #Create Teacher model graph
                    ##=========================================================##
                    cnn3dT.make_cnn_model(*model_params.get_args_for_arch())
                    ##=========================================================##

            # No explicit device assignment for the rest. Because trained has piecewise_constant that is only on cpu, and so is saver.        
            with tf.variable_scope("trainer"):
                self._log.print3("=========== Building Trainer ===========\n")
            
                trainer = Trainer( *( self._params.get_args_for_trainer() + [cnn3d] + [cnn3dT] ) )
                trainer.create_optimizer( *self._params.get_args_for_optimizer() ) # Trainer and net connect here.      
            #with tf.variable_scope("trainer_tch")
                ##===========================================================================##
                trainerT = Trainer( *( self._params.get_args_for_trainer() + [cnn3dT] + [cnn3d] ) )
                trainerT.create_optimizer( *self._params.get_args_for_optimizer() )
                ##===========================================================================##

            # The below should not create any new tf.variables.
            self._log.print3("=========== Compiling the Training Function ===========")
            self._log.print3("=======================================================\n")
            cnn3d.setup_ops_n_feeds_to_train( self._log,
                                              trainer.get_total_cost(),
                                              trainer.get_param_updates_for_stu_and_tch_model() # list of ops
                                            )

            cnn3dT.setup_ops_n_feeds_to_train( self._log,
                                              trainerT.get_total_cost(),
                                              trainerT.get_param_updates_for_stu_and_tch_model() # list of ops
                                            )
            
            
            self._log.print3("=========== Compiling the Validation Function =========")
            cnn3d.setup_ops_n_feeds_to_val( self._log )
            
            cnn3dT.setup_ops_n_feeds_to_val( self._log )

            self._log.print3("=========== Compiling the Testing Function ============")
            cnn3d.setup_ops_n_feeds_to_test( self._log,
                                             self._params.indices_fms_per_pathtype_per_layer_to_save ) # For validation with full segmentation
            
            cnn3dT.setup_ops_n_feeds_to_testT( self._log,
                                             self._params.indices_fms_per_pathtype_per_layer_to_save ) # For validation with full segmentation
            
            # Create the savers
            saver_all = tf.train.Saver() # Will be used during training for saving everything.
            collection_vars_net = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="net") # Alternative: tf.train.Saver([v for v in tf.all_variables() if v.name.startswith("net"])
            collection_vars_tch_net = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="tch_net") #Get variables in teacher model
            

            collection_vars_trainer = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="trainer")
            #collection_vars_trainer_tch = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="trainer_tch")
            
            
        #self._print_vars_in_collection(collection_vars_net, "net")
        self._print_vars_in_collection(collection_vars_tch_net, "tch_net")
        
        with tf.Session( graph=graphTf, config=tf.ConfigProto(log_device_placement=False, device_count={'CPU':999, 'GPU':99}) ) as sessionTf:
            # Load or initialize parameters
            file_to_load_params_from = self._params.get_path_to_load_model_from()
            if file_to_load_params_from is not None: # Load params
                self._log.print3("=========== Loading parameters from specified saved model ===============")
                chkpt_fname = tf.train.latest_checkpoint( file_to_load_params_from ) if os.path.isdir( file_to_load_params_from ) else file_to_load_params_from
                ##============================================Added2019.10.05================================================##
                self._log.print3("\n===============================MT-WA Method=================================\n")
                reader = tf.train.NewCheckpointReader(chkpt_fname)
                var = reader.get_variable_to_shape_map() #type:dict, every element type: list
                #self._print_vars_in_collection(var, "net")
                l = len(collection_vars_net) // 2
                rdmInit_vars = collection_vars_net[l-11:] + collection_vars_tch_net[l-11:] ## softmax layer parameters.and also random initialize the plus units weights.
                tf.variables_initializer(rdmInit_vars).run() 

                ##================================Mannually initialize widened params==========================================##
                ## first to deal with the first layer parameters since the number of input channels are different
                ## here we use the average value of four channels
                for key in var:
                    if (key == 'net/W') or (key == 'tch_net/W'):
                        temp = reader.get_tensor(key)
                        W0 = tf.reshape(tf.reduce_mean(temp, axis=1), shape=[temp.shape[0], 1, temp.shape[2], temp.shape[3], temp.shape[4]])
                    if (key == 'net/b') or (key == 'tch_net/b'):
                        temp = reader.get_tensor(key)
                        b0 = tf.reshape(tf.reduce_mean(temp, axis=0), shape=[1,])
                    if (key == 'net/W_8') or (key == 'tch_net/W_8'):
                        temp = reader.get_tensor(key)
                        W8 = tf.reshape(tf.reduce_mean(temp, axis=1), shape=[temp.shape[0], 1, temp.shape[2], temp.shape[3], temp.shape[4]])
                    if (key == 'net/b_1') or (key == 'tch_net/b_1'):
                        temp = reader.get_tensor(key)
                        b1 = tf.reshape(tf.reduce_mean(temp, axis=0), shape=[1,])
                    if (key == 'net/W_16') or (key == 'tch_net/W_16'):
                        temp = reader.get_tensor(key)
                        W16 = tf.reshape(tf.reduce_mean(temp, axis=1), shape=[temp.shape[0], 1, temp.shape[2], temp.shape[3], temp.shape[4]])
                    if (key == 'net/b_2') or (key == 'tch_net/b_2'):
                        temp = reader.get_tensor(key)
                        b2 = tf.reshape(tf.reduce_mean(temp, axis=0), shape=[1,])
                    ##############################################################################################
                    if (key == 'net/W_25') or (key == 'tch_net/W_25'):  #########################Not Sure!!!!!!!!!!! And delete the corresponding W25 in stroke model variables.
                        W25 = reader.get_tensor(key)
                        sessionTf.run(tf.assign(collection_vars_net[l-11][:250, :, :, :, :], W25))
                        sessionTf.run(tf.assign(collection_vars_tch_net[l-11][:250, :, :, :, :], W25))
                    if (key == 'net/gBn_23') or (key == 'tch_net/gBn_23'):
                        gBn23 = reader.get_tensor(key)
                        sessionTf.run(tf.assign(collection_vars_net[l-9][:250], gBn23))
                        sessionTf.run(tf.assign(collection_vars_tch_net[l-9][:250], gBn23))
                    if (key == 'net/bBn_23') or (key == 'tch_net/bBn_23'):
                        bBn23 = reader.get_tensor(key)
                        sessionTf.run(tf.assign(collection_vars_net[l-8][:250], bBn23))
                        sessionTf.run(tf.assign(collection_vars_tch_net[l-8][:250], bBn23))
                    if (key == 'net/muBnsForRollingAverage_23') or (key == 'tch_net/muBnsForRollingAverage_23'):
                        mu23 = reader.get_tensor(key)
                        sessionTf.run(tf.assign(collection_vars_net[l-7][:, :250], mu23))
                        sessionTf.run(tf.assign(collection_vars_tch_net[l-7][:, :250], mu23))
                    if (key == 'net/varBnsForRollingAverage_23') or (key == 'tch_net/varBnsForRollingAverage_23'):
                        var23 = reader.get_tensor(key)
                        sessionTf.run(tf.assign(collection_vars_net[l-6][:, :250], var23))
                        sessionTf.run(tf.assign(collection_vars_tch_net[l-6][:, :250], var23))
                    if (key == 'net/sharedNewMu_B_23') or (key == 'tch_net/sharedNewMu_B_23'):
                        sMu23 = reader.get_tensor(key)
                        sessionTf.run(tf.assign(collection_vars_net[l-5][:250], sMu23))
                        sessionTf.run(tf.assign(collection_vars_tch_net[l-5][:250], sMu23))
                    if (key == 'net/sharedNewVar_B_23') or (key == 'tch_net/sharedNewVar_B_23'):
                        sVar23 = reader.get_tensor(key)
                        sessionTf.run(tf.assign(collection_vars_net[l-4][:250], sVar23))
                        sessionTf.run(tf.assign(collection_vars_tch_net[l-4][:250], sVar23))
                    if (key == 'net/aPrelu_23') or (key == 'tch_net/aPrelu_23'):
                        aP23 = reader.get_tensor(key)
                        sessionTf.run(tf.assign(collection_vars_net[l-3][:250], aP23))
                        sessionTf.run(tf.assign(collection_vars_tch_net[l-3][:250], aP23))

                ##Delete them widened params from the var list, random initialize the rest in the list#####
                i = 0
                while i < (len(collection_vars_net) - 6):
                    v = collection_vars_net[i]
                    #self._log.print3(str(v))
                    
                    if v.name == 'net/W:0':
                        sessionTf.run(tf.assign(v, W0))
                        sessionTf.run(tf.assign(collection_vars_tch_net[i], W0))

                        self._log.print3(str(collection_vars_net[i].shape)+str(collection_vars_tch_net[i].shape))
                        del collection_vars_net[i]
                        del collection_vars_tch_net[i]
                        i = i - 1
                    if v.name == 'net/b:0':
                        sessionTf.run(tf.assign(v, b0))
                        sessionTf.run(tf.assign(collection_vars_tch_net[i], b0))

                        self._log.print3(str(collection_vars_net[i].shape))
                        del collection_vars_net[i]
                        del collection_vars_tch_net[i]
                        i = i - 1
                        
                    if v.name == 'net/W_8:0':
                        sessionTf.run(tf.assign(v, W8))
                        sessionTf.run(tf.assign(collection_vars_tch_net[i], W8))

                        self._log.print3(str(collection_vars_net[i].shape))
                        del collection_vars_net[i]
                        del collection_vars_tch_net[i]
                        i = i - 1
                    if v.name == 'net/b_1:0':
                        sessionTf.run(tf.assign(v, b1))
                        sessionTf.run(tf.assign(collection_vars_tch_net[i], b1))

                        self._log.print3(str(collection_vars_net[i].shape))
                        del collection_vars_net[i]
                        del collection_vars_tch_net[i]
                        i = i - 1
                        
                    if v.name == 'net/W_16:0':
                        sessionTf.run(tf.assign(v, W16))
                        sessionTf.run(tf.assign(collection_vars_tch_net[i], W16))
                        self._log.print3(str(collection_vars_net[i].shape))
                        del collection_vars_net[i]
                        del collection_vars_tch_net[i]
                        i = i - 1
                    if v.name == 'net/b_2:0':
                        sessionTf.run(tf.assign(v, b2))
                        sessionTf.run(tf.assign(collection_vars_tch_net[i], b2))

                        self._log.print3(str(collection_vars_net[i].shape))
                        del collection_vars_net[i]
                        del collection_vars_tch_net[i]
                        i = i - 1
                    i = i + 1

                reset_trainer = True
                saver_net = tf.train.Saver( var_list = collection_vars_net[:l-11] + collection_vars_tch_net
                [:l-11] ) # Used to load the net's parameters.
                saver_trainer = tf.train.Saver( var_list = collection_vars_trainer ) # Used to load the trainer's parameters.
    
                ##============================================================================================================##
                self._log.print3("Loading checkpoint file:" + str(chkpt_fname))
                self._log.print3("Loading network parameters...")
                try:
                    saver_net.restore(sessionTf, chkpt_fname)
                    self._log.print3("Network parameters were loaded.")
                except Exception as e: handle_exception_tf_restore(self._log, e)
                
                if not reset_trainer:
                    self._log.print3("Loading trainer parameters...")
                    saver_trainer.restore(sessionTf, chkpt_fname)
                    self._log.print3("Trainer parameters were loaded.")
                else:
                    self._log.print3("Reset of trainer parameters was requested. Re-initializing them...")
                    tf.variables_initializer(var_list = collection_vars_trainer).run()
                    self._log.print3("Trainer parameters re-initialized.")
            else :
                self._log.print3("=========== Initializing network and trainer variables  ===============")
                # tf.variables_initializer(var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) ).run() # Initializes all.
                # Initialize separate as below, so that in case I miss a variable, I will get an error and I will know.
                tf.variables_initializer(var_list = collection_vars_net).run()
                tf.variables_initializer(var_list = collection_vars_trainer).run()
                self._log.print3("All variables were initialized.")
                
                filename_to_save_with = self._params.filepath_to_save_models + ".initial." + datetimeNowAsStr()
                self._log.print3("Saving the initial model at:" + str(filename_to_save_with))
                saver_all.save( sessionTf, filename_to_save_with+".model.ckpt", write_meta_graph=False )
                # tf.train.write_graph( graph_or_graph_def=sessionTf.graph.as_graph_def(), logdir="", name=filename_to_save_with+".graph.pb", as_text=False)
             
            self._log.print3("")
            self._log.print3("=======================================================")
            self._log.print3("============== Training the CNN model =================")
            self._log.print3("=======================================================\n")
            
            do_training( *( [sessionTf, saver_all, cnn3d, cnn3dT, trainer, trainerT] + self._params.get_args_for_train_routine() ) )
            
            # Save the trained model.
            filename_to_save_with = self._params.filepath_to_save_models + ".final." + datetimeNowAsStr()
            self._log.print3("Saving the final model at:" + str(filename_to_save_with))
            saver_all.save( sessionTf, filename_to_save_with+".model.ckpt", write_meta_graph=False )
            
            
        self._log.print3("\n=======================================================")
        self._log.print3("=========== Training session finished =================")
        self._log.print3("=======================================================")
        
