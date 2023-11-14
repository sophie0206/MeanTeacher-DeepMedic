# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division
import os

from deepmedicO.frontEnd.session import Session
from deepmedicO.frontEnd.configParsing.utils import getAbsPathEvenIfRelativeIsGiven
from deepmedicO.frontEnd.configParsing.testSessionParams import TestSessionParameters
from deepmedicO.frontEnd.sessHelpers import makeFoldersNeededForTestingSession, handle_exception_tf_restore

from deepmedicO.neuralnet.cnn3d import Cnn3d
from deepmedicO.routines.testingDELTA import performInferenceOnWholeVolumes

import tensorflow as tf

class TestSession(Session):
    
    def __init__(self, cfg):
        self._out_folder_preds = None
        self._out_folder_fms = None
        
        Session.__init__(self, cfg)
        
    def _make_sess_name(self):
        sess_name = TestSessionParameters.getSessionName( self._cfg[self._cfg.SESSION_NAME] )
        return sess_name
    
    def make_output_folders(self):
        [self._log_folder_abs,
         self._out_folder_preds,
         self._out_folder_fms] = makeFoldersNeededForTestingSession( self._main_out_folder_abs, self._sess_name )
         
         
    def compile_session_params_from_cfg(self, *args):
        (model_params,) = args
        
        self._params = TestSessionParameters(
                                    log = self._log,
                                    mainOutputAbsFolder = self._main_out_folder_abs,
                                    folderForPredictions = self._out_folder_preds,
                                    folderForFeatures = self._out_folder_fms,
                                    num_classes = model_params.numberClasses,
                                    cfg = self._cfg )
        
        self._log.print3("")
        self._log.print3("============     NEW TESTING SESSION    ===============")
        self._params.print_params()    
        self._log.print3("=======================================================\n")
        
        return self._params
        
        
    def _ask_user_if_test_with_random(self):
        user_input = None
        try:
            user_input = raw_input("WARN:\t Testing was requested, but without specifying a pretrained, saved model to load.\n"+\
                                   "\t A saved model can be specified via the command line or the test-config file.\n" +\
                                   "\t Please see documentation or run ./deepmedicORun -h for help on how to load a model.\n"+\
                                   "\t Do you wish to continue and test inference with a randomly initialized model? [y/n] : ")
            while user_input not in ['y','n']: 
                user_input = raw_input("Please specify 'y' or 'n': ")
        except:
            print("\nERROR:\tTesting was requested, but without specifying a pretrained, saved model to load."+\
                  "\n\tTried to ask for user input whether to continue testing with a randomly initialized model, but failed."+\
                  "\n\tReason unknown (nohup? remote?)."+\
                  "\n\tPlease see documentation or run ./deepmedicORun -h for help on how to load a model."+\
                  "\n\tExiting."); exit(1)
        if user_input == 'y':
            pass
        else:
            print("Exiting as requested."); exit(0)
    
    
    def run_session(self, *args):
        (sess_device,
         model_params,) = args
        
        graphTf = tf.Graph()
        
        with graphTf.as_default():
            with graphTf.device(sess_device): # Throws an error if GPU is specified but not available.
                self._log.print3("=========== Making the CNN graph... ===============")
                cnn3d = Cnn3d()
                with tf.variable_scope("net"):
                    cnn3d.make_cnn_model( *model_params.get_args_for_arch() ) # Creates the network's graph (without optimizer).
                    
            
            cnn3d.setup_ops_n_feeds_to_test( self._log,
                                             self._params.indices_fms_per_pathtype_per_layer_to_save )
            # Create the saver
            saver_all = tf.train.Saver() # saver_net would suffice
            collection_vars_net = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="net") # Alternative: tf.train.Saver([v for v in tf.all_variables() if v.name.startswith("net"])  type:list          
            #print(collection_vars_net)
            #collection_vars_trainer = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="trainer")
            
        with tf.Session( graph=graphTf, config=tf.ConfigProto(log_device_placement=False, device_count={'CPU':999, 'GPU':99}) ) as sessionTf:
            file_to_load_params_from = self._params.get_path_to_load_model_from()
            if file_to_load_params_from is not None: # Load params
                self._log.print3("=========== Loading parameters from specified saved model ===============")
                chkpt_fname = tf.train.latest_checkpoint( file_to_load_params_from ) if os.path.isdir( file_to_load_params_from ) else file_to_load_params_from
                self._log.print3("Loading parameters from:" + str(chkpt_fname))
                
                reader = tf.train.NewCheckpointReader(chkpt_fname)
                var = reader.get_variable_to_shape_map() #type:dict, every element type: list
                #self._print_vars_in_collection(var, "net")
                
                tf.variables_initializer(collection_vars_net[-2:]).run()
                
                self._log.print3("Average of 4 Channel training Baseline. Classic Fine-tune, no augmentation.\n")
                for key in var:
                    if key == 'net/W_26':
                        W26 = reader.get_tensor(key)
                        #print("QQQQ")
                        #W26 = tf.tile(reader.get_tensor(key), [2,1,1,1,1])
                    '''
                    if key == 'net/W':
                        tmp = reader.get_tensor(key)
                        #print("QQQQ")
                        W0 = tf.reshape(tf.reduce_mean(temp, axis=1), shape=[temp.shape[0], 1, temp.shape[2], temp.shape[3], temp.shape[4]])
                    if key == 'net/b':
                        temp = reader.get_tensor(key)
                        b0 = tf.reshape(tf.reduce_mean(temp, axis=0), shape=[1,])
                        print(b0)
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
                        b2 = tf.reshape(tf.reduce_mean(temp, axis=0), shape=[1,])'''
                    
                self._log.print3("First Part")       
                i = 0
                ops = []
                while i < (len(collection_vars_net)):
                    v = collection_vars_net[i]
                    #self._log.print3(str(v))
                    if v.name == 'net/W_26:0':
                        ops.append(tf.assign(v, tf.concat([W26, W26], axis=0)))
                        self._log.print3(str(collection_vars_net[i].shape))
                        #del collection_vars_net[i]
                        #i = i - 1
                    '''    
                    if v.name == 'net/W:0':
                        ops.append(tf.assign(v, W0))
                        self._log.print3(str(collection_vars_net[i].shape))
                        del collection_vars_net[i]
                        i = i - 1
                    if v.name == 'net/b:0':
                        ops.append(tf.assign(v, b0))
                        self._log.print3(str(collection_vars_net[i].shape))
                        del collection_vars_net[i]
                        i = i - 1
                        
                    if v.name == 'net/W_8:0':
                        ops.append(tf.assign(v, W8))
                        self._log.print3(str(collection_vars_net[i].shape))
                        del collection_vars_net[i]
                        i = i - 1
                    if v.name == 'net/b_1:0':
                        ops.append(tf.assign(v, b1))
                        self._log.print3(str(collection_vars_net[i].shape))
                        del collection_vars_net[i]
                        i = i - 1
                        
                    if v.name == 'net/W_16:0':
                        ops.append(tf.assign(v, W16))
                        self._log.print3(str(collection_vars_net[i].shape))
                        del collection_vars_net[i]
                        i = i - 1
                    if v.name == 'net/b_2:0':
                        ops.append(tf.assign(v, b2))
                        self._log.print3(str(collection_vars_net[i].shape))
                        del collection_vars_net[i]
                        i = i - 1'''
                    i = i + 1
                
                self._log.print3("Second Part")  
                sessionTf.run(ops)
                
                #reset_trainer = True
                #self._print_vars_in_collection(collection_vars_net, "net")
                
                
                saver_net = tf.train.Saver( var_list=collection_vars_net[:-2] ) # Used to load the net's parameters.
                #saver_trainer = tf.train.Saver( var_list=collection_vars_trainer ) # Used to load the trainer's parameters.
                ##=========================================================================================================================##
                
                try:
                    saver_net.restore(sessionTf, chkpt_fname)
                    self._log.print3("Parameters were loaded.")
                except Exception as e: handle_exception_tf_restore(self._log, e)
                
            else:
                self._ask_user_if_test_with_random() # Asks user whether to continue with randomly initialized model. It exits if no is given.
                self._log.print3("")
                self._log.print3("=========== Initializing network variables  ===============")
                tf.variables_initializer( var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="net") ).run()
                self._log.print3("Model variables were initialized.")
                
                
            self._log.print3("")
            self._log.print3("======================================================")
            self._log.print3("=========== Testing with the CNN model ===============")
            self._log.print3("======================================================\n")
            
            performInferenceOnWholeVolumes( *( [sessionTf, cnn3d] + self._params.get_args_for_testing() + [ saver_net, chkpt_fname] ))
        
        self._log.print3("")
        self._log.print3("======================================================")
        self._log.print3("=========== Testing session finished =================")
        self._log.print3("======================================================")
