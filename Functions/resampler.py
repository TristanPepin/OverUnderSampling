# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 13:55:47 2021

@author: peptr
"""
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, SMOTEN, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE
from imblearn.under_sampling import RandomUnderSampler
from Functions.Generative_models.GAN import CTGAN
from table_evaluator import TableEvaluator
import numpy as np
import pandas as pd

class Resampler():
    
    def __init__(self,method = 'random over', alpha = None, **kwargs):
        self.kwargs = kwargs
        self.method = method
        self.alpha = alpha

        self.define_resampler()
        self.fitted = False
        
        
    def define_resampler(self):
        
        sampling_strategy = 1.0 if self.alpha is None else self.alpha
        self.under = False
        
        if 'RANDOM' in self.method.upper() :
            if 'UNDER' in self.method.upper() :
                self.resampler = RandomUnderSampler(sampling_strategy = sampling_strategy,**self.kwargs)
                self.under = True
            else :
                self.resampler = RandomOverSampler(sampling_strategy = sampling_strategy,**self.kwargs)
        elif 'SMOTE' in self.method.upper() :
            if 'K' in self.method.upper() :
                self.resampler = KMeansSMOTE(sampling_strategy = sampling_strategy,**self.kwargs)
            elif 'B' in self.method.upper() :
                self.resampler = BorderlineSMOTE(sampling_strategy = sampling_strategy,**self.kwargs)
            elif 'V' in self.method.upper() :
                self.resampler = SVMSMOTE(sampling_strategy = sampling_strategy,**self.kwargs)
            elif 'N' in self.method.upper() :
                self.resampler = SMOTEN(sampling_strategy = sampling_strategy,**self.kwargs)
            else :
                self.resampler = SMOTE(sampling_strategy = sampling_strategy,**self.kwargs)
        elif 'ADASYN' in self.method.upper() :
            self.resampler = ADASYN(sampling_strategy = sampling_strategy,**self.kwargs)

        elif 'GAN' in self.method.upper() :
            print(self.kwargs)
            self.resampler = CTGAN(sampling_strategy = sampling_strategy,**self.kwargs)
            
        else : 
            raise NameError('Resampler not implemented yet.')
            
            
    def set_sampling_strategy(self,
                              alpha : float):
        self.alpha = alpha
        self.define_resampler()
        
        
    def return_pre_fit_index(self):
        
        if not self.fitted :
            return None
        return self.index
        
        
    def fit(self,X,y):
        
        count = y.value_counts().sort_values(ascending = True)
        self.minority_class = count.index[0]
        self.colnames = X.columns
        self.index = X.index
        self.target = y.name
        self.len_original = len(y)
        self.fitted = True
        
        
    def fit_resample(self,X,y):
        
        if not self.fitted:
            self.fit(X,y)
        
        if 'GAN' not in self.method.upper() :
            X,y = np.array(X), np.array(y)

        X_resampled, y_resampled = self.resampler.fit_resample(X,y)
        return pd.DataFrame(X_resampled,columns = self.colnames).reset_index(drop=True), pd.Series(y_resampled,name=self.target).reset_index(drop=True)
        
        
    def plot_statistics(self, X_resampled, y_resampled):
        
        if not self.fitted:
            raise NameError('Resampler not fitted yet') 
        if self.under : 
            raise NameError('Not implemented yet for under sampling')
        
        original_X_data = X_resampled[X_resampled.index < self.len_original]
        original_y_data = y_resampled[y_resampled.index < self.len_original]
        
        real = original_X_data[original_y_data == self.minority_class]
        fake = X_resampled[X_resampled.index >= self.len_original]

        table_evaluator = TableEvaluator(real, fake)
        table_evaluator.visual_evaluation()