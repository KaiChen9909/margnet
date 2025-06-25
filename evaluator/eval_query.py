import numpy as np
import os 
import tempfile
import shutil
import random
# from TabDDPM.scripts.sample import sample
from copy import deepcopy
from pathlib import Path
from evaluator.data.dataset import read_pure_data
from evaluator.data.data_utils import *
# from DP_MERF.sample import merf_heterogeneous_sample

def query_succeed(x: np.array, query_attr, query, query_type):
    query_res = np.full(len(x), True)
    for i in range(len(query)):
        if query_type[i] == 'num':
            query_res = query_res & (x[:, query_attr[i]].astype(float) >= query[i][0]) & (x[:, query_attr[i]].astype(float) <= query[i][1])
        elif query_type[i] == 'cat':
            query_res = query_res & (x[:, query_attr[i]] == query[i][0])
    return sum(query_res)

def make_query(
    synthetic_data_path,
    data_path,
    task_type,
    query_times,
    attr_num,
    seeds = 0,
    test_data = 'test'
):
    random.seed(seeds)
    print("-" * 100)
    print('Starting query error evaluation')

    if test_data =='real':
        X_num_real, X_cat_real, y_real = read_pure_data(data_path, split = 'train')
    else:
        X_num_real, X_cat_real, y_real = read_pure_data(data_path, split = 'test')
        
    data_list = [X_num_real, X_cat_real, y_real.reshape(-1,1).astype(int) if y_real is not None else y_real]
    data_list = [arr for arr in data_list if arr is not None]

    if len(data_list) > 1:
        real_data = np.concatenate(data_list, axis=1)
    elif len(data_list) == 1:
        real_data = data_list[0]
    else:
        raise ValueError('Invalid Real Data')

    X_num_fake, X_cat_fake, y_fake = read_pure_data(synthetic_data_path, split = 'train')
    data_list = [X_num_fake, X_cat_fake, y_fake.reshape(-1,1).astype(int) if y_fake is not None else y_fake]
    data_list = [arr for arr in data_list if arr is not None]

    if len(data_list) > 1:
        fake_data = np.concatenate(data_list, axis=1)
    elif len(data_list) == 1:
        fake_data = data_list[0]
    else:
        raise ValueError('Invalid Synthetic Data')
    

    # obtain the domain of variables
    num_attr = 0
    cat_attr = 0
    y_attr = 0
    num_range = None
    cat_range = None
    y_range = None
    if X_num_real is not None: 
        num_attr = X_num_real.shape[1]
        num_range = get_numerical_range(X_num_real)
    if X_cat_real is not None: 
        cat_attr = X_cat_real.shape[1]
        cat_range = get_category_range(X_cat_real)
    if y_real is not None:
        y_attr = 1
        y_range = get_numerical_range(y_real.reshape(-1,1)) if task_type == 'regression' else get_category_range(y_real.reshape(-1,1))

    error = []
    for i in range(query_times):
        query_attr = np.random.choice(np.arange(0, num_attr + cat_attr + y_attr), size = attr_num, replace = False)
        query = []
        query_type = []
        
        for x in query_attr:
            if x < num_attr:
                query.append(sorted([random.uniform(num_range[x][0], num_range[x][1]), random.uniform(num_range[x][0], num_range[x][1])]))
                query_type.append('num')
            elif x >= num_attr and x < (num_attr + cat_attr): 
                query.append(np.random.choice(cat_range[x - num_attr], 1, replace=False))
                query_type.append('cat')
            else:
                if task_type == 'regression':
                    query.append(sorted([random.uniform(y_range[0][0], y_range[0][1]), random.uniform(y_range[0][0], y_range[0][1])]))
                    query_type.append('num')
                else:
                    query.append(np.random.choice(y_range[0], 1, replace=False))
                    query_type.append('cat')
            
            error.append(abs(
                query_succeed(real_data, query_attr, query, query_type)/len(real_data) - 
                query_succeed(fake_data, query_attr, query, query_type)/len(fake_data)
            ))
    
    print('query error:', np.mean(error))
    return np.mean(error)

        
