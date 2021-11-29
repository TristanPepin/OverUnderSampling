# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 13:37:27 2021

@author: peptr
"""

import pandas as pd
import os
import sys
from ctgan import CTGANSynthesizer
import json
from table_evaluator import TableEvaluator
import numpy as np
import torch

class CTGAN():
    
    def __init__(self, 
                 sampling_strategy = 1,
                 path_save_model = None,
                 model_name = None,
                 embedding_dim=128, 
                 generator_dim=(256, 256), 
                 discriminator_dim=(256, 256),
                 generator_lr=2e-4, 
                 generator_decay=1e-6, 
                 discriminator_lr=2e-4,
                 discriminator_decay=1e-6, 
                 batch_size=500, 
                 discriminator_steps=1,
                 log_frequency=True, 
                 verbose=True, 
                 epochs=300, 
                 pac=10, 
                 cuda=True) : 

        self.sampling_strategy = sampling_strategy
        self.path_save_model = path_save_model
        self.mode_name = model_name
        
        self.embedding_dim = embedding_dim
        self.generator_dim = generator_dim
        self.discriminator_dim = discriminator_dim

        self.generator_lr = generator_lr
        self.generator_decay = generator_decay
        self.discriminator_lr = discriminator_lr
        self.discriminator_decay = discriminator_decay

        self.batch_size = batch_size
        self.discriminator_steps = discriminator_steps
        self.log_frequency = log_frequency
        self.verbose = verbose
        self.epochs = epochs
        self.pac = pac
        
        self.cuda = cuda
        self.categorical_variables = None
        self.fitted = False


        self.GAN = CTGANSynthesizer(embedding_dim = self.embedding_dim, 
                                    generator_dim = self.generator_dim, 
                                    discriminator_dim = self.discriminator_dim,
                                    generator_lr = self.generator_lr, 
                                    generator_decay = self.generator_decay, 
                                    discriminator_lr = self.discriminator_lr,
                                    discriminator_decay = self.discriminator_decay, 
                                    batch_size = self.batch_size, 
                                    discriminator_steps = self.discriminator_steps,
                                    log_frequency = self.log_frequency, 
                                    verbose = self.verbose, 
                                    epochs = self.epochs, 
                                    pac = self.pac, 
                                    cuda = self.cuda)
        

        
    def reset(self):
        self.GAN = CTGANSynthesizer(embedding_dim = self.embedding_dim, 
                                    generator_dim = self.generator_dim, 
                                    discriminator_dim = self.discriminator_dim,
                                    generator_lr = self.generator_lr, 
                                    generator_decay = self.generator_decay, 
                                    discriminator_lr = self.discriminator_lr,
                                    discriminator_decay = self.discriminator_decay, 
                                    batch_size = self.batch_size, 
                                    discriminator_steps = self.discriminator_steps,
                                    log_frequency = self.log_frequency, 
                                    verbose = self.verbose, 
                                    epochs = self.epochs, 
                                    pac = self.pac, 
                                    cuda = self.cuda)

    def set_categorical_variables(self,categorical : list):
        self.categorical_variables = categorical


        
        
    def fit(self,X,y):

        quantitative = X.columns.to_list()
        categorical = []

        if self.categorical_variables is not None : 
            for cat in self.categorical_variables :
                categorical.append(cat)
                quantitative.remove(cat)

        # Finding the class to be sampled
        count = y.value_counts().sort_values(ascending = True)
        self.minority_class = count.index[0]
        self.n_minority_class = count[count.index[0]]
        self.n_majority_class = count[count.index[1]]

        X_target_category = X.loc[y == self.minority_class]
        self.GAN.fit(X_target_category,categorical)
        self.n_generate = int(self.sampling_strategy*self.n_majority_class - self.n_minority_class)

        self.fitted = True
        if self.path_save_model is not None :
            if not os.path.exists(self.path_save_model):
                os.mkdir(self.path_save_model)

            if self.model_name is not None :
                torch.save(self.GAN,self.path_save_model + '/' + self.model_name + '_gan') 
            else :
                torch.save(self.GAN,self.path_save_model + '/_gan') 

    
    def fit_resample(self,X,y):

        if not self.fitted:
           self.fit(X,y)

        fake = self.GAN.sample(self.n_generate)
        generated_y = np.array([self.minority_class for i in range(self.n_generate)])

        return np.concatenate((X.values,fake)), np.concatenate((y.values,generated_y))
        
        
