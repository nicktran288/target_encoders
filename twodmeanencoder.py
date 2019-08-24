

import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


class TwoDMeanEncoder(object):
    
    
    def __init__(self, col1, col2, encoding_name=None, n_folds=5, 
                 shuffle=False, random_state=None, min_train_samples=1, drop=True, drop_ref_cols=True):
        
        self.col1 = col1
        self.col2 = col2
        
        self.col1_agg_name = '{}_mean'.format(self.col1)
        self.col2_agg_name = '{}_{}_mean'.format(self.col1, self.col2)
        
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.random_state = random_state
        self.min_train_samples = min_train_samples
        
        self.drop = drop
        self.drop_ref_cols = drop_ref_cols

        if isinstance(encoding_name, str):
            self.encoding_name = encoding_name
        elif encoding_name == None:
            self.encoding_name = None
        else:
            raise TypeError('encoding_name expects str or None (not "{}").'.format(type(encoding_name)))
        

        
    def _drop_cols(self, data):
        
        if self.drop:
            data.drop([self.col1, self.col2], axis=1, inplace=True)
        if self.drop_ref_cols:
            data.drop([self.col1_agg_name, self.col2_agg_name], axis=1, inplace=True)
            if 'fold' in data.columns:
                data.drop('fold', axis=1, inplace=True)
            

            
    def _calculate_fold_encodings(self, data, target, min_samples=1):
        
        data = data.copy()
        
        reference_tables = {}
        
        col1_agg = data[[self.col1, target]].groupby(self.col1).mean().reset_index().rename(columns={target:self.col1_agg_name})
        col1_col2_agg = data[[self.col1, self.col2, target]].groupby([self.col1, self.col2]).mean().reset_index().rename(columns={target:self.col2_agg_name})
        
        class_encodings = col1_col2_agg.merge(col1_agg, how='left', on=self.col1)
        
        class_encodings[self.encoding_name] = (class_encodings[self.col2_agg_name] - class_encodings[self.col1_agg_name]) / class_encodings[self.col1_agg_name]
        
        return class_encodings
            
            

    def transform_train(self, X, y):
        
        X = X.copy()
        
        if isinstance(y, str):
            
            if y in X.columns:
                self.target_class = y
                data = X.copy()
                
                self.drop_target=False
                
            else:
                raise KeyError('"{}" is not present in X.\ny must be col name in X, or a Series.'.format(y))
                
        elif isinstance(y, pd.Series):
            
            if y.name in X.columns:
                warnings.warn('y feature name {} detected in X.  Will use existing X column.'.format(y.name))
                
                self.target_class = y.name
                data = X.copy()
                
                self.drop_target=False
                
            elif (X.sort_index().index == y.sort_index().index).all():
                self.target_class = y.name
                data = pd.concat([X.copy(), y.copy()], axis=1)
                
                self.drop_target=True
                
            else:
                raise IndexError('X and y must have identical length and index values.')
        else:
            raise TypeError('transform_train expects DataFrame for X and Series or col name for y.')
                

        if self.encoding_name == None:
            prefix = self.target_class
            suffix = '{}_by_{}'.format(self.col1, self.col2)
            
            self.encoding_name = '{}_encoding_{}'.format(prefix, suffix)
            
        self.train_reference_tables = []
        encoded_data = []
        
        kf = KFold(self.n_folds, shuffle=self.shuffle, random_state=self.random_state)
        
        for n, (train_index, val_index) in enumerate(kf.split(data)):
            
            train_data = data.iloc[train_index]
            val_data = data.iloc[val_index]

            reference_table = self._calculate_fold_encodings(train_data, 
                                                         target=self.target_class,
                                                         min_samples=self.min_train_samples)
            
            val_data = val_data.reset_index().merge(reference_table, how='left', on=[self.col1, self.col2]).set_index('index')
            
            self.train_reference_tables.append({'fold': n, 'index': val_data.index, 'reference_tables': reference_table})
            
            val_data['fold'] = n

            encoded_data.append(val_data)
        
        data_out = pd.concat(encoded_data)

        
        self.test_reference_tables = data_out[[self.col1, self.col2, self.encoding_name]].groupby([self.col1, self.col2]).mean().reset_index()

        self._drop_cols(data_out)
        
        if self.drop_target:
            data.drop(self.target_class, axis=1, inplace=True)
  
        return data_out
    
    
    
    def transform_test(self, X, y=None):
        
        X = X.copy()
        
        data_out = X.reset_index().merge(self.test_reference_tables, how='left', on=[self.col1, self.col2]).set_index('index')
        
        self._drop_cols(data_out)
        
        return data_out
        
        


