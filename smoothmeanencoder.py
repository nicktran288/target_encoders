
import warnings

import numpy as np
import pandas as pd



class SmoothMeanEncoder(object):
    
    def __init__(self, feature_cols, encoding_name=None, C=100):
        
        if isinstance(feature_cols, str):
            self.feature_cols = [feature_cols]
        elif isinstance(feature_cols, list):
            self.feature_cols = feature_cols
        else:
            raise TypeError('feature_cols expects str or list (not "{}") of column(s).'.format(type(feature_cols)))
            
        if isinstance(encoding_name, str):
            self.encoding_name = encoding_name
        elif encoding_name == None:
            self.encoding_name = None
        else:
            raise TypeError('encoding_name expects str or None (not "{}").'.format(type(encoding_name)))
        
        self.C = C
        
        
        print('\nfeature_cols expects a list of columns ordered from the highest to lowest level in the class hierarchy.')
        print('\nSmoothMeanEncoder is initialized with the following class hierarchy:\n')
        indent = ''
        for level, feature in enumerate(self.feature_cols):
            indent = '  ' * level
            print('{}> {}'.format(indent, feature))


            
    def _get_smooth_means(self, data, hierarchy, target, C, encoding_name):

        n = len(hierarchy)

        if n == 1:
            means = data[hierarchy + [target]].groupby(hierarchy).agg(['mean', 'count']).reset_index(col_level=1)
            means.columns = means.columns.get_level_values(1)

            global_mean = data[target].mean()

            means[encoding_name] = (means['count'] * means['mean'] + C * global_mean) / (means['count'] + C)

            return means[hierarchy + [encoding_name]]

        else:
            means = data[hierarchy + [target]].groupby(hierarchy).agg(['mean', 'count']).reset_index(col_level=1)
            means.columns = means.columns.get_level_values(1)

            upper_means = self._get_smooth_means(data, hierarchy[:n-1], 'age_in_weeks', C, encoding_name)
            means = means.merge(upper_means, how='left', on=hierarchy[:n-1])

            means[encoding_name] = (means['count'] * means['mean'] + C * means[encoding_name]) / (means['count'] + C)

            return means[hierarchy + [encoding_name]]


        
    def fit(self, X, y):
        
        if isinstance(y, str):
            if y in X.columns:
                self.target = y
                data = X.copy()
            else:
                raise KeyError('"{}" is not present in X.\ny must be col name in X, or a Series.'.format(y))
                
        elif isinstance(y, pd.Series):
            if y.name in X.columns:
                warnings.warn('y feature name "{}" detected in X.  Will use existing X column.'.format(y.name))
                
                self.target = y.name
                data = X.copy()
                
            elif (X.sort_index().index == y.sort_index().index).all():
                self.target = y.name
                data = pd.concat([X.copy(), y.copy()], axis=1)
                
            else:
                raise IndexError('X and y must have identical length and index values.')
                
        else:
            raise TypeError('fit expects DataFrame for X and Series or col name for y.')
            

        if self.encoding_name == None:
            prefix = self.target
            suffix = '_'.join([str(col_name).replace(' ', '') for col_name in self.feature_cols])
            
            self.encoding_name = prefix + '_encoding_' + suffix
        
        self.reference_table = self._get_smooth_means(data, self.feature_cols, self.target, self.C, self.encoding_name)
        
        return self
        
        
        
    def transform(self, X, y=None):
        
        data = X.copy()
        
        data = data.merge(self.reference_table, how='left', on=self.feature_cols)
        
        return data
    
    
    
    def fit_transform(self, X, y):
        
        return self.fit(X, y).transform(X)
    
    