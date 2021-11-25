# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 12:07:56 2021

@author: peptr
"""

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.base import clone
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd


class GridSearchClassRatio():
    
    def __init__(self,model,k=5,method = 'Stratified'):
        self.model = model
        self.k = k
        self.method = method
        self.result_dict_ = dict()
        self.best_alpha_ = dict()
        self.best_score_ = dict()
        self.best_resampler_ = dict()
        
        
        
    def perform_k_fold_cross_validation(self,model,resampler,X,y,metric='f1_score',
                                    return_scores = False):
        
    
        # Choice of cross validation method, stratified or not
        if self.method.upper() == 'STRATIFIED': 
            kf = StratifiedKFold(n_splits=self.k)
        else : 
            kf = KFold(n_splits=self.k)
            
        # Metric Choice : 
        if 'RECALL' in metric.upper():
            m = recall_score
        elif 'ACCURACY' in metric.upper():
            m = accuracy_score
        elif 'PRECISION' in metric.upper():
            m = precision_score
        else :
            m = f1_score
            
        X_cols = X.columns
        target = y.name
        
        X,y = np.array(X), np.array(y)
    
        kf.get_n_splits(X, y)
        scores = []
    
        
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            df_X_train = pd.DataFrame(X_train,columns=X_cols)
            df_y_train = pd.Series(y_train,name=target)
            resampler.fit(df_X_train,df_y_train)
            Xr, yr = resampler.fit_resample(df_X_train,df_y_train)
            
            model_cv = clone(model)
            model_cv.fit(Xr,yr)
            scores.append(m(y_test,model_cv.predict(X_test)))
            
        if return_scores :
            return scores
            
        return np.mean(scores), np.std(scores)
        

    def perform_grid_search_class_ratio(self,resampler,X,y,range_alpha = [0.25, 0.5, 0.75],
                                        metric='f1_score'):
        
        if metric not in self.result_dict_.keys():
            self.result_dict_[metric] = dict()
            self.best_alpha_[metric] = range_alpha[0]
            self.best_score_[metric] = 0
            self.best_resampler_[metric] = resampler
        
        bar = tqdm(range_alpha)
        for alpha in bar :
            bar.set_description(str(alpha))
            resampler.set_sampling_strategy(alpha)
            scores = self.perform_k_fold_cross_validation(self.model, resampler, X, y, metric = metric, return_scores = True)
            self.result_dict_[metric][str(alpha)] = scores
            
            if np.mean(scores) > self.best_score_[metric] :
                self.best_score_[metric] = np.mean(scores)
                self.best_alpha_[metric] = alpha
                self.best_resampler_[metric] = resampler
                
            
        return self.best_resampler_[metric]
        
        
        
    def plot_results(self,plot_type = 'boxplot'):
        
        if plot_type == 'boxplot':
        
            for metric in self.result_dict_.keys() :
                result_dict = self.result_dict_[metric]
                fig,ax = plt.subplots(figsize=(10,5))
                ax.boxplot(result_dict.values())
                ax.set_xticklabels(result_dict.keys())
                ax.set_ylabel('Scores ({})'.format(metric))
                plt.show()
        else :
            plt.figure(figsize=(10,5))
            for metric in self.result_dict_.keys() :
                alpha_range = np.array(list(self.result_dict_[metric].keys()),dtype=float)
                values = np.mean(list(self.result_dict_[metric].values()),axis=1)
                plt.scatter(alpha_range,values,label=metric)
                
            plt.legend()      
            plt.xlabel('Class Ratio')
            plt.ylabel('Metric, {}-kfold cv'.format(self.k))
            plt.show()
            
        
    
   

