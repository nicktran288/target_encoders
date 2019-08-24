

# Author: Wah (Nick) Tran
# Status: Production


import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


class CVTargetEncoder(object):
    '''
    Performs target encoding using cross validation method.  Allows for classes defined by multiple categorical
    variables and regularization using min_samples for calculating encoding for a class.

    For each of k folds, encodes using out-of-fold target means.  If out-of-fold counts for a class is below 
    min_train_samples (or class is not present), then if a class hierarchy is in place the mean for the next 
    highest level is used, otherwise the global out-of-fold target mean is used.

    Encoding for test sets is performed using the same method except the encodings for the train set is used
    as the target.
    '''
    
    
    def __init__(self, feature_cols, encoding_name=None, n_folds=5, shuffle=False, random_state=None,
                 min_train_samples=1, drop=True, drop_ref_cols=True):
        '''
        Parameters
        ----------

        feature_cols : list or string
            Column(s) defining encoding class.
            If list of length > 1, columns should be ordered from highest level of class hierarchy to lowest.

        encoding_name : string (default=None)
            Name of column containing encoded values.
            If None, default column name will be generated using feature_cols and target variable name.

        n_folds : int (default=5)
            Number of training folds over which to calculate target means.

        shuffle : bool (default=False)
            If True, shuffle train set before creating splits.

        random_state : int (default=None)
            Random seed to reproduce splits if shuffle is set to True.

        min_train_samples : int (default=1)
            Minimum number of out-of-fold samples a class must have to generate an encoding.

        drop : bool (default=True)
            If True, original categorical columns used to define encoding class will be dropped.

        drop_ref_cols : bool (default=True)
            If True, reference columns containing means for each level in the encoded class hierarchy will be dropped


        Attributes
        ----------

        mean_cols : list
            List of column names for intermediate columns containing mean values.

        target_class : str
            Name of column containing target values used for encoding.
        '''
        
        if isinstance(feature_cols, str):
            self.feature_cols = [feature_cols]
        elif isinstance(feature_cols, list):
            self.feature_cols = feature_cols
        else:
            raise TypeError('feature_cols expects str or list (not "{}") of column(s).'.format(type(feature_cols)))

        self.mean_cols = [x + '_mean' for x in self.feature_cols]
            
        if isinstance(encoding_name, str):
            self.encoding_name = encoding_name
        elif encoding_name == None:
            self.encoding_name = None
        else:
            raise TypeError('encoding_name expects str or None (not "{}").'.format(type(encoding_name)))
        
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.random_state = random_state
        self.min_train_samples = min_train_samples
        self.drop = drop
        self.drop_ref_cols = drop_ref_cols
        

        
        print('\nfeature_cols expects a list of columns ordered from the highest to lowest level in the class hierarchy.')
        print('\nMeanEncoder is initialized with the following class hierarchy:\n')
        indent = ''
        for level, feature in enumerate(self.feature_cols):
            indent = '  ' * level
            print('{}> {}'.format(indent, feature))

            
    def _drop_cols(self, data):
        if self.drop:
            data.drop(self.feature_cols, axis=1, inplace=True)
        if self.drop_ref_cols:
            data.drop(self.mean_cols + ['fold_global_mean'], axis=1, inplace=True)
            
        
    def _calculate_fold_means(self, train_data, target, min_samples=1):
        
        train_data = train_data.copy()
        
        reference_tables = {}
        
        hierarchy = []
        for n, (class_col, mean_col) in enumerate(zip(self.feature_cols, self.mean_cols)):
            
            hierarchy.append(class_col)
            
            class_means = train_data[hierarchy + [target]].groupby(hierarchy).agg(['mean', 'count']).reset_index(col_level=1)
            class_means.columns = class_means.columns.get_level_values(1)
            
            class_means = class_means[class_means['count'] >= min_samples].drop('count', axis=1).reset_index(drop=True)
            class_means = class_means.rename(columns={'mean': mean_col})
            
            reference_tables[class_col] = class_means
        
        fold_global_mean = train_data[target].mean()
        
        reference_tables['fold_global_mean'] = fold_global_mean

        return reference_tables
    
    
    
    def _merge_reference_tables(self, val_data, reference_tables):
        
        hierarchy = []
        
        for class_col, mean_col in zip(self.feature_cols, self.mean_cols):
            hierarchy.append(class_col)
            
            class_means = reference_tables[class_col]

            val_data = val_data.reset_index().merge(class_means, how='left', on=hierarchy).set_index('index')

            
        val_data['fold_global_mean'] = reference_tables['fold_global_mean']
        
        return val_data
        
    
    
    def _create_encoding(self, val_data):
        
        val_data[self.encoding_name] = np.nan
        
        for mean_col in self.mean_cols[::-1]:
            val_data[self.encoding_name].fillna(val_data[mean_col], inplace=True)
        
        val_data[self.encoding_name].fillna(val_data['fold_global_mean'], inplace=True)
        
        return val_data
            
            

    def transform_train(self, X, y):
        
        X = X.copy()
        
        if isinstance(y, str):
            if y in X.columns:
                self.target_class = y
                data = X.copy()
            else:
                raise KeyError('"{}" is not present in X.\ny must be col name in X, or a Series.'.format(y))
        elif isinstance(y, pd.Series):
            if y.name in X.columns:
                warnings.warn('y feature name {} detected in X.  Will use existing X column.'.format(y.name))
                
                self.target_class = y.name
                data = X.copy()
                
            elif (X.sort_index().index == y.sort_index().index).all():
                self.target_class = y.name
                data = pd.concat([X.copy(), y.copy()], axis=1)
                
            else:
                raise IndexError('X and y must have identical length and index values.')
        else:
            raise TypeError('transform_train expects DataFrame for X and Series or col name for y.')
                
            
            
        
#         y = y.copy()
#         data = pd.concat([X, y], axis=1)
        
        data['fold'] = np.nan
        
#         self.target_class = [y.name]
        
        if self.encoding_name == None:
            prefix = self.target_class
            suffix = '_'.join([str(col_name).replace(' ', '') for col_name in self.feature_cols])
            
            self.encoding_name = prefix + '_encoding_' + suffix
            
        self.train_reference_tables = []
        
        encoded_data = []
        
        kf = KFold(self.n_folds, shuffle=self.shuffle, random_state=self.random_state)
        
        for n, (train_index, val_index) in enumerate(kf.split(data)):
            
            train_data = data.iloc[train_index]
            val_data = data.iloc[val_index]

            reference_table = self._calculate_fold_means(train_data, 
                                                         target=self.target_class,
                                                         min_samples=self.min_train_samples)
            
            val_data = self._merge_reference_tables(val_data, reference_table)
            
            self.train_reference_tables.append({'fold': n, 'index': val_data.index, 'reference_tables': reference_table})
            val_data['fold'] = n

            encoded_data.append(val_data)
        
        data_out = pd.concat(encoded_data)
        
        data_out = self._create_encoding(data_out)
        
        self.test_reference_tables = self._calculate_fold_means(data_out, self.encoding_name)
        
        self._drop_cols(data_out)
        data.drop(self.target_class, axis=1, inplace=True)
  
        return data_out
    
    
    def transform_test(self, X, y=None):
        
        X = X.copy()
        
        data_out = self._create_encoding(self._merge_reference_tables(X, self.test_reference_tables))
        
        self._drop_cols(data_out)
        
        return data_out
        
        


