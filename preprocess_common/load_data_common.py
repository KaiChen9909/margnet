import numpy as np 
import pandas as pd
import os
import json
import math
import sklearn.preprocessing
from preprocess_common.preprocess import * 
from util.rho_cdp import cdp_rho

class data_preporcesser_common():
    def __init__(self, args):
        self.num_encoder = None
        self.cat_encoder = None
        self.num_col = 0
        self.cat_col = 0
        self.y_col = 0
        self.args = args
        self.cat_output_type = 'one_hot' if self.args.method in ['merf', 'merf_fit'] else 'ordinal'

        if self.args.dataset in ['gauss10', 'gauss30', 'gauss50'] and self.args.method in ['merf', 'ddpm', 'ctgan']: self.split_case = True
        else: self.split_case = False

    def load_data(self, path, rho, rate=0.1):
        # preprocesser and column info will be saved in this class
        # dataframe, domain domain, portion of rho will be returned

        if self.split_case:
            return self.load_data_split(path, rho, rate=rate)

        num_prep = self.args.num_preprocess
        rare_threshold = self.args.rare_threshold
        print(f'Numerical discretizer is {num_prep}')

        X_num = None 
        X_cat = None 
        y = None

        if os.path.exists(os.path.join(path, 'X_num_train.npy')):
            X_num = np.load(os.path.join(path, 'X_num_train.npy'), allow_pickle=True)
        if os.path.exists(os.path.join(path, 'X_cat_train.npy')):
            X_cat = np.load(os.path.join(path, 'X_cat_train.npy'), allow_pickle=True)
        if os.path.exists(os.path.join(path, 'y_train.npy')):   
            y = np.load(os.path.join(path, 'y_train.npy'), allow_pickle=True)
        
        
        num_divide, cat_divide = calculate_rho_allocate(X_num, X_cat, num_prep)
    
        if X_num is not None:
            if num_prep != 'none':
                ord = False if self.args.method in ['rap', 'gsd'] else True
                self.num_encoder = discretizer(num_prep, num_divide * rate * rho, ord=ord)
            else:
                if self.args.method == 'ctgan':
                    self.num_encoder = sklearn.preprocessing.MinMaxScaler(feature_range=(-100,100))    
                else:
                    self.num_encoder = sklearn.preprocessing.MinMaxScaler(feature_range=(0,1))
            X_num = self.num_encoder.fit_transform(X_num)
            self.num_col = X_num.shape[1]
        if X_cat is not None:
            self.cat_encoder = rare_merger(cat_divide * rate * rho, rare_threshold=rare_threshold, output_type=self.cat_output_type)
            X_cat = self.cat_encoder.fit_transform(X_cat)
            self.cat_col = X_cat.shape[1]
        if y is not None:
            self.y_col = 1

        if self.args.method in ('merf', 'merf_fit', 'ddpm', 'pe_ddpm'):
            df = {
                "X_num": X_num,
                "X_cat": X_cat,
                "y": y
            }
            domain = {
                "X_num": [len(set(X_num[:, i])) for i in range(self.num_col)] if X_num is not None else [],
                "X_cat": [len(cat) for cat in self.cat_encoder.ordinal_encoder.categories_] if X_cat is not None else [],
                "y": [len(set(y))] if y is not None else []
            }
        else:
            col_name = [f'num_attr_{i}' for i in range(1, self.num_col + 1)] + [f'cat_attr_{i}' for i in range(1, self.cat_col + 1)]
            if y is not None:
                col_name += ['y_attr']

            data_list = [X_num, X_cat, y.reshape(-1,1) if y is not None else y]
            data_list = [arr for arr in data_list if arr is not None]

            if len(data_list) > 1:
                df = pd.DataFrame(np.concatenate(data_list, axis=1), columns=col_name)
            elif len(data_list) == 1:
                df = pd.DataFrame(data_list[0], columns=col_name)
            else:
                raise ValueError('Invalid Data Input')

            domain = json.load(open(os.path.join(path, 'domain.json')))
            for i in range(1, self.num_col + 1):
                domain[f'num_attr_{i}'] = min(domain[f'num_attr_{i}'], len(set(X_num[:, i-1])))
            for i in range(1, self.cat_col + 1):
                domain[f'cat_attr_{i}'] = min(domain[f'cat_attr_{i}'], len(set(X_cat[:, i-1]))) 

        print('preprocessed data domain:', domain)
        print('finish loading data')
        print('-'*100)
        return df, domain, rate*(num_divide + cat_divide)
    

    def load_pub_data(self, path):
        if self.split_case:
            return None
        
        X_num = None 
        X_cat = None 
        y = None

        if os.path.exists(os.path.join(path, 'X_num_pretrain.npy')):
            X_num = np.load(os.path.join(path, 'X_num_pretrain.npy'), allow_pickle=True)
        if os.path.exists(os.path.join(path, 'X_cat_pretrain.npy')):
            X_cat = np.load(os.path.join(path, 'X_cat_pretrain.npy'), allow_pickle=True)
        if os.path.exists(os.path.join(path, 'y_pretrain.npy')):
            y = np.load(os.path.join(path, 'y_pretrain.npy'), allow_pickle=True)
        

        if X_num is not None:
            X_num = self.num_encoder.transform(X_num)
        if X_cat is not None:
            X_cat = self.cat_encoder.transform(X_cat)        

        if self.args.method in ('merf', 'ddpm', 'pe_ddpm'):
            if X_num is None and X_cat is None and y is None:
                df = None
            else:
                df = {
                    "X_num": X_num,
                    "X_cat": X_cat,
                    "y": y
                }
        else:
            col_name = [f'num_attr_{i}' for i in range(1, self.num_col + 1)] + [f'cat_attr_{i}' for i in range(1, self.cat_col + 1)]
            if y is not None:
                col_name += ['y_attr']

            data_list = [X_num, X_cat, y.reshape(-1,1) if y is not None else y]
            data_list = [arr for arr in data_list if arr is not None]

            if len(data_list) > 1:
                df = pd.DataFrame(np.concatenate(data_list, axis=1), columns=col_name)
            elif len(data_list) == 1:
                df = pd.DataFrame(data_list[0], columns=col_name)
            else:
                df = None
        
        return df

    def reverse_data(self, df, path=None):
        if isinstance(df, pd.DataFrame):
            df = df.to_numpy()
        
        if self.split_case:
            return self.reverse_data_split(df, path=path)
        
        x_num = None 
        x_cat = None 
        y = None 
        
        if self.num_col > 0:
            x_num = df[:, 0:self.num_col]
            if x_num.ndim == 1:
                x_num = x_num.reshape(-1,1)
            
            if self.num_encoder is not None:
                x_num = self.num_encoder.inverse_transform(x_num).astype(float)
            if path is not None:
                np.save(os.path.join(path, 'X_num_train.npy'), x_num)

        if self.cat_col > 0:
            x_cat = df[:, self.num_col:self.num_col+self.cat_col]
            if x_cat.ndim == 1:
                x_cat = x_cat.reshape(-1,1)
            
            if self.cat_encoder is not None:
                x_cat = self.cat_encoder.inverse_transform(x_cat).astype(str)
            if path is not None:
                np.save(os.path.join(path, 'X_cat_train.npy'), x_cat) 
        
        if self.y_col > 0:
            y = df[:, -1].reshape(-1).astype(int)
            if path is not None:
                np.save(os.path.join(path, 'y_train.npy'), y) 
        
        return x_num, x_cat, y
    

    def load_data_split(self, path, rho, rate=0.1):
        # preprocesser and column info will be saved in this class
        # dataframe, domain domain, portion of rho will be returned
        num_prep = self.args.num_preprocess
        print(f'Numerical discretizer is {num_prep}')
        print(f'label discretizer is Kbins')

        X_num = None 
        X_cat = None 
        y = None

        X_num = np.load(os.path.join(path, 'X_num_train.npy'), allow_pickle=True)
        
        X_num = X_num[:,:-1]
        y = X_num[:,-1]
        print(y.shape)
        
        num_divide, cat_divide = 0, 0

        self.num_encoder = sklearn.preprocessing.MinMaxScaler(feature_range=(0,1))
        X_num = self.num_encoder.fit_transform(X_num)
        self.num_col = X_num.shape[1]
        self.cat_col = 0
        
        if self.args.method == 'ddpm':
            self.y_encoder = sklearn.preprocessing.MinMaxScaler(feature_range=(0,1))
            y = self.y_encoder.fit_transform(y.reshape(-1,1)).reshape(-1)
        else:
            self.y_encoder = discretizer('uniform_kbins', num_divide * rate * rho, bin_number=10, ord=True)
            y = self.y_encoder.fit_transform(y.reshape(-1,1)).reshape(-1)

        if self.args.method in ('merf', 'ddpm'):
            df = {
                "X_num": X_num,
                "X_cat": X_cat,
                "y": y
            }
            domain = {
                "X_num": [len(set(X_num[:, i])) for i in range(self.num_col)] if X_num is not None else [],
                "X_cat": [len(cat) for cat in self.cat_encoder.ordinal_encoder.categories_] if X_cat is not None else [],
                "y": [len(set(y))] if y is not None else []
            }
        else:
            col_name = [f'num_attr_{i}' for i in range(1, self.num_col + 1)] + [f'cat_attr_{i}' for i in range(1, self.cat_col + 1)]
            if y is not None:
                col_name += ['y_attr']

            data_list = [X_num, X_cat, y.reshape(-1,1) if y is not None else y]
            data_list = [arr for arr in data_list if arr is not None]

            if len(data_list) > 1:
                df = pd.DataFrame(np.concatenate(data_list, axis=1), columns=col_name)
            elif len(data_list) == 1:
                df = pd.DataFrame(data_list[0], columns=col_name)
            else:
                raise ValueError('Invalid Data Input')

            domain = json.load(open(os.path.join(path, 'domain.json')))
            for i in range(1, self.num_col + 1):
                domain[f'num_attr_{i}'] = min(domain[f'num_attr_{i}'], len(set(X_num[:, i-1])))
            for i in range(1, self.cat_col + 1):
                domain[f'cat_attr_{i}'] = min(domain[f'cat_attr_{i}'], len(set(X_cat[:, i-1]))) 
        

        print('preprocessed data domain:', domain)
        print('finish loading data')
        print('-'*100)
        return df, domain, rate*(num_divide + cat_divide)


    def reverse_data_split(self, df, path=None):
        x_num = None 
        x_cat = None 
        y = None 

        x_num = df[:, 0:self.num_col]
        if x_num.ndim == 1:
            x_num = x_num.reshape(-1,1)
        if self.num_encoder is not None:
            x_num = self.num_encoder.inverse_transform(x_num).astype(float)
        

        y = df[:, -1].astype(int).reshape(-1,1)
        y = self.y_encoder.inverse_transform(y).astype(float)

        x_num = np.concatenate((x_num, y), axis=1)
        y = None

        if path is not None:
            np.save(os.path.join(path, 'X_num_train.npy'), x_num)
        
        return x_num, x_cat, y