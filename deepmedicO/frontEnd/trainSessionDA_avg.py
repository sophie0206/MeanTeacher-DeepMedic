# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

# For Average Initialization Depthth-augmented Method!!!!!!!!!!!


from __future__ import absolute_import, print_function, division
import os

from deepmedicO.frontEnd.session import Session
from deepmedicO.frontEnd.configParsing.utils import getAbsPathEvenIfRelativeIsGiven
from deepmedicO.frontEnd.configParsing.trainSessionParams import TrainSessionParameters
from deepmedicO.frontEnd.sessHelpers import makeFoldersNeededForTrainingSession, handle_exception_tf_restore

from deepmedicO.logging.utils import datetimeNowAsStr
from deepmedicO.neuralnet.cnn3dDA import Cnn3d
from deepmedicO.neuralnet.trainer import Trainer

from deepmedicO.routines.training import do_training

import tensorflow as tf
import numpy as np

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
                with tf.variable_scope("net"):
                    cnn3d.make_cnn_model( *model_params.get_args_for_arch() )
                    # I have now created the CNN graph. But not yet the Optimizer's graph.
            
            # No explicit device assignment for the rest. Because trained has piecewise_constant that is only on cpu, and so is saver.        
            with tf.variable_scope("trainer"):
                self._log.print3("=========== Building Trainer ===========\n")
                trainer = Trainer( *( self._params.get_args_for_trainer() + [cnn3d] ) )
                trainer.create_optimizer( *self._params.get_args_for_optimizer() ) # Trainer and net connect here.
                
            # The below should not create any new tf.variables.
            self._log.print3("=========== Compiling the Training Function ===========")
            self._log.print3("=======================================================\n")
            cnn3d.setup_ops_n_feeds_to_train( self._log,
                                              trainer.get_total_cost(),
                                              trainer.get_param_updates_wrt_total_cost() # list of ops
                                            )
            
            self._log.print3("=========== Compiling the Validation Function =========")
            cnn3d.setup_ops_n_feeds_to_val( self._log )
            
            self._log.print3("=========== Compiling the Testing Function ============")
            cnn3d.setup_ops_n_feeds_to_test( self._log,
                                             self._params.indices_fms_per_pathtype_per_layer_to_save ) # For validation with full segmentation
            
            # Create the savers
            saver_all = tf.train.Saver() # Will be used during training for saving everything.
            collection_vars_net = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="net") # Alternative: tf.train.Saver([v for v in tf.all_variables() if v.name.startswith("net"])  type:list          
            
            collection_vars_trainer = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="trainer")
            
        #self._print_vars_in_collection(collection_vars_net, "net")
        #self._print_vars_in_collection(collection_vars_trainer, "trainer")
        
        with tf.Session( graph=graphTf, config=tf.ConfigProto(log_device_placement=False, device_count={'CPU':999, 'GPU':99}, gpu_options=tf.GPUOptions(allow_growth=True)) ) as sessionTf:
            # Load or initialize parameters
            file_to_load_params_from = self._params.get_path_to_load_model_from()
            if file_to_load_params_from is not None: # Load params
                self._log.print3("=========== Loading parameters from specified saved model ===============")
                chkpt_fname = tf.train.latest_checkpoint( file_to_load_params_from ) if os.path.isdir( file_to_load_params_from ) else file_to_load_params_from
                
                ##===============================================================================================================##
                reader = tf.train.NewCheckpointReader(chkpt_fname)
                var = reader.get_variable_to_shape_map() #type:dict, every element type: list
                #self._print_vars_in_collection(var, "net")
                rdmInit_vars = collection_vars_net[-11:] ## get the last two, softmax layer parameters.and also random initialize the plus units weights.
                tf.variables_initializer(rdmInit_vars).run()
                
                self._log.print3("Average Initialization training\nDepth-augmented Method!")
                '''
                for key in var:
                    if key == 'net/W':
                        temp = reader.get_tensor(key)
                        W0 = tf.reshape(tf.reduce_mean(temp, axis=1), shape=[temp.shape[0], 1, temp.shape[2], temp.shape[3], temp.shape[4]])
                    if key == 'net/b':
                        temp = reader.get_tensor(key)
                        b0 = tf.reshape(tf.reduce_mean(temp, axis=0), shape=[1,])
                    if key == 'net/W_8':
                        temp = reader.get_tensor(key)
                        W8 = tf.reshape(tf.reduce_mean(temp, axis=1), shape=[temp.shape[0], 1, temp.shape[2], temp.shape[3], temp.shape[4]])
                    if key == 'net/b_1':
                        temp = reader.get_tensor(key)
                        b1 = tf.reshape(tf.reduce_mean(temp, axis=0), shape=[1,])
                    if key == 'net/W_16':
                        temp = reader.get_tensor(key)
                        W16 = tf.reshape(tf.reduce_mean(temp, axis=1), shape=[temp.shape[0], 1, temp.shape[2], temp.shape[3], temp.shape[4]])
                    if key == 'net/b_2':
                        temp = reader.get_tensor(key)
                        b2 = tf.reshape(tf.reduce_mean(temp, axis=0), shape=[1,])
                    
                        
                i = 0
                while i < (len(collection_vars_net) - 6):
                    v = collection_vars_net[i]
                    #self._log.print3(str(v))
                    
                    if v.name == 'net/W:0':
                        sessionTf.run(tf.assign(v, W0))
                        self._log.print3(str(collection_vars_net[i].shape))
                        del collection_vars_net[i]
                        i = i - 1
                    if v.name == 'net/b:0':
                        sessionTf.run(tf.assign(v, b0))
                        self._log.print3(str(collection_vars_net[i].shape))
                        del collection_vars_net[i]
                        i = i - 1
                        
                    if v.name == 'net/W_8:0':
                        sessionTf.run(tf.assign(v, W8))
                        self._log.print3(str(collection_vars_net[i].shape))
                        del collection_vars_net[i]
                        i = i - 1
                    if v.name == 'net/b_1:0':
                        sessionTf.run(tf.assign(v, b1))
                        self._log.print3(str(collection_vars_net[i].shape))
                        del collection_vars_net[i]
                        i = i - 1
                        
                    if v.name == 'net/W_16:0':
                        sessionTf.run(tf.assign(v, W16))
                        self._log.print3(str(collection_vars_net[i].shape))
                        del collection_vars_net[i]
                        i = i - 1
                    if v.name == 'net/b_2:0':
                        sessionTf.run(tf.assign(v, b2))
                        self._log.print3(str(collection_vars_net[i].shape))
                        del collection_vars_net[i]
                        i = i - 1
                    i = i + 1
                '''
                reset_trainer = True
                #self._print_vars_in_collection(collection_vars_net[:-10], "net")
                saver_net = tf.train.Saver( var_list=collection_vars_net[:-11] ) # Used to load the net's parameters.
                saver_trainer = tf.train.Saver( var_list=collection_vars_trainer ) # Used to load the trainer's parameters.
                ##=========================================================================================================================##
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
            
            do_training( *( [sessionTf, saver_all, cnn3d, trainer] + self._params.get_args_for_train_routine() ) )
            
            # Save the trained model.
            filename_to_save_with = self._params.filepath_to_save_models + ".final." + datetimeNowAsStr()
            self._log.print3("Saving the final model at:" + str(filename_to_save_with))
            saver_all.save( sessionTf, filename_to_save_with+".model.ckpt", write_meta_graph=False )
            
            
        self._log.print3("\n=======================================================")
        self._log.print3("=========== Training session finished =================")
        self._log.print3("=======================================================")
        
