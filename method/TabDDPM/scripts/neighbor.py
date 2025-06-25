import numpy as np
import pandas as pd
import sklearn
from copy import deepcopy
from scipy.spatial.distance import cdist
from scipy.spatial.distance import cosine

def prepare_dataset(temp_dataset):
    # scale x_num
    num_encoder = sklearn.preprocessing.MinMaxScaler(
            feature_range=(0, 1)
        )
    temp_dataset.X_num['train'] = num_encoder.fit_transform(temp_dataset.X_num['train'])
    temp_dataset.X_num['pretrain'] = num_encoder.transform(temp_dataset.X_num['pretrain'])


    # one-hot encoding x_cat
    cat_encoder = sklearn.preprocessing.OneHotEncoder(
            handle_unknown='ignore', sparse_output=False, dtype=np.float32
        )
    temp_dataset.X_cat['train'] = cat_encoder.fit_transform(temp_dataset.X_cat['train'])
    temp_dataset.X_cat['pretrain'] = cat_encoder.transform(temp_dataset.X_cat['pretrain'])


    # scale/one-hot encoding y
    if temp_dataset.task_type == 'regression':
        y_encoder = sklearn.preprocessing.MinMaxScaler(
                feature_range=(0, 1)
            )
    else:
        y_encoder = sklearn.preprocessing.OneHotEncoder(
                handle_unknown='ignore', sparse_output=False, dtype=np.float32
            )
    temp_dataset.y['train'] = y_encoder.fit_transform(temp_dataset.y['train'].reshape(-1,1))
    temp_dataset.y['pretrain'] = y_encoder.transform(temp_dataset.y['pretrain'].reshape(-1,1))


    return num_encoder, cat_encoder, y_encoder


def count_nearest_neighbors(train_array, pretrain_array):
    distances = cdist(train_array, pretrain_array, metric='cityblock')
    nearest_indices = np.argmin(distances, axis=1)
    
    count_list = [0] * len(pretrain_array)
    for idx in nearest_indices:
        count_list[idx] += 1
    
    return np.array(count_list, dtype=np.float64)


def count_nearest_neighbors_cosine(train_array, pretrain_array):
    similarities = np.zeros(len(pretrain_array))
    
    for i, query_vector in enumerate(train_array):
        cosine_similarities = 1 - np.array([cosine(query_vector, target_vector) for target_vector in pretrain_array])
        max_idx = np.argmax(cosine_similarities)
        similarities[max_idx] += cosine_similarities[max_idx]
    
    return similarities



def neighbor_sample(dataset, rho, **kwargs):
    temp_dataset = deepcopy(dataset)
    prepare_dataset(temp_dataset)

    pretrain_array = np.hstack((temp_dataset.X_num['pretrain'], temp_dataset.X_cat['pretrain'], temp_dataset.y['pretrain']))
    train_array = np.hstack((temp_dataset.X_num['train'], temp_dataset.X_cat['train'], temp_dataset.y['train']))

    count = count_nearest_neighbors(train_array, pretrain_array)
    count += np.sqrt(1/(2*rho)) * np.random.normal(size = count.shape[0]) # sensitivity = 1
    count = np.clip(count, a_min=0, a_max=None)

    # sample
    int_part = np.floor(count).astype(int)
    frac_part = count - int_part

    additional_sample = np.random.rand(len(count)) < frac_part
    final_counts = int_part + additional_sample.astype(int)
    indices = np.repeat(np.arange(len(count)), final_counts)

    dataset.X_num['pretrain'] = dataset.X_num['pretrain'][indices]
    dataset.X_cat['pretrain'] = dataset.X_cat['pretrain'][indices]
    dataset.y['pretrain'] = dataset.y['pretrain'][indices]


def neighbor_freq_sample(dataset, rho, size=None, **kwargs):
    temp_dataset = deepcopy(dataset)
    prepare_dataset(temp_dataset)

    pretrain_array = np.hstack((temp_dataset.X_num['pretrain'], temp_dataset.X_cat['pretrain'], temp_dataset.y['pretrain']))
    train_array = np.hstack((temp_dataset.X_num['train'], temp_dataset.X_cat['train'], temp_dataset.y['train']))

    count = count_nearest_neighbors(train_array, pretrain_array)
    count += np.sqrt(1/(2*rho)) * np.random.normal(size = count.shape[0]) # sensitivity = 1
    count = np.clip(count, a_min=0, a_max=None)
    count = count/np.sum(count)

    # sample
    if size is None: size = len(count)
    indices = np.random.choice(np.arange(len(count)), size=size, p=count, replace=True)

    dataset.X_num['pretrain'] = dataset.X_num['pretrain'][indices]
    dataset.X_cat['pretrain'] = dataset.X_cat['pretrain'][indices]
    dataset.y['pretrain'] = dataset.y['pretrain'][indices]


def neighbor_cosine_sample(dataset, rho, size=None, **kwargs):
    temp_dataset = deepcopy(dataset)
    prepare_dataset(temp_dataset)

    pretrain_array = np.hstack((temp_dataset.X_num['pretrain'], temp_dataset.X_cat['pretrain'], temp_dataset.y['pretrain']))
    train_array = np.hstack((temp_dataset.X_num['train'], temp_dataset.X_cat['train'], temp_dataset.y['train']))

    count = count_nearest_neighbors_cosine(train_array, pretrain_array)
    count += np.sqrt(1/(2*rho)) * np.random.normal(size = count.shape[0]) # sensitivity = 1
    count = np.clip(count, a_min=0, a_max=None)
    count = count/np.sum(count)

    # sample
    if size is None: size = len(count)
    indices = np.random.choice(np.arange(len(count)), size=size, p=count, replace=True)

    dataset.X_num['pretrain'] = dataset.X_num['pretrain'][indices]
    dataset.X_cat['pretrain'] = dataset.X_cat['pretrain'][indices]
    dataset.y['pretrain'] = dataset.y['pretrain'][indices]
