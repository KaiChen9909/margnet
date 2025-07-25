import numpy as np
import os
import sys
target_path="./"
sys.path.append(target_path)

from evaluator.data.data_utils import *
from evaluator.data.dataset import *
from sklearn.utils import shuffle
import random 
from pathlib import Path
from pprint import pprint
from evaluator.data.metrics import MetricsReport
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import classification_report, r2_score

def train_catboost(
    parent_dir,
    data_path,
    eval_type,
    T_dict,
    seed = 0,
    model_name = None,
    params = None,
    change_val = True,
    device = None, # dummy
    model_step = 'finetune',
    test_data = 'test'
):
    random.seed(seed)
    if eval_type != "real":
        synthetic_data_path = os.path.join(parent_dir)
    info = load_json(os.path.join(data_path, 'info.json'))
    T = Transformations(**T_dict) 

    print('start to read data')
    
    if change_val:
        X_num_real, X_cat_real, y_real, X_num_val, X_cat_val, y_val = read_changed_val(data_path, val_size=0.2, model_step=model_step, seed=seed)

    X = None
    print('-'*100)
    if eval_type == 'merged':
        print('loading merged data...')
        if not change_val:
            X_num_real, X_cat_real, y_real = read_pure_data(data_path, 'train' if model_step=='finetune' else 'pretrain')
        X_num_fake, X_cat_fake, y_fake = read_pure_data(synthetic_data_path, 'train' if model_step=='finetune' else 'pretrain')

        ###
        # dists = privacy_metrics(data_path, synthetic_data_path)
        # bad_fakes = dists.argsort()[:int(0.25 * len(y_fake))]
        # X_num_fake = np.delete(X_num_fake, bad_fakes, axis=0)
        # X_cat_fake = np.delete(X_cat_fake, bad_fakes, axis=0) if X_cat_fake is not None else None
        # y_fake = np.delete(y_fake, bad_fakes, axis=0)
        ###

        y = np.concatenate([y_real, y_fake], axis=0)

        X_num = None
        if X_num_real is not None:
            X_num = np.concatenate([X_num_real, X_num_fake], axis=0)

        X_cat = None
        if X_cat_real is not None:
            X_cat = np.concatenate([X_cat_real, X_cat_fake], axis=0)

    elif eval_type == 'synthetic':
        print(f'loading synthetic data: {parent_dir}')
        X_num, X_cat, y = read_pure_data(synthetic_data_path, 'train' if model_step=='finetune' else 'pretrain')

    elif eval_type == 'real':
        print('loading real data...')
        if not change_val:
            X_num, X_cat, y = read_pure_data(data_path, 'train' if model_step=='finetune' else 'pretrain')
        else: 
            X_num, X_cat, y = X_num_real, X_cat_real, y_real
    else:
        raise "Choose eval method"

    if not change_val:
        X_num_val, X_cat_val, y_val = read_pure_data(data_path, 'val' if model_step=='finetune' else 'preval')
    
    if test_data == 'real':
        X_num_test, X_cat_test, y_test = read_pure_data(data_path, 'train')
    else:
        X_num_test, X_cat_test, y_test = read_pure_data(data_path, 'test')

    D = Dataset(
        {'train': X_num, 'val': X_num_val, 'test': X_num_test} if X_num is not None else None,
        {'train': X_cat, 'val': X_cat_val, 'test': X_cat_test} if X_cat is not None else None,
        {'train': y, 'val': y_val, 'test': y_test},
        {},
        TaskType(info['task_type']),
        info.get('n_classes')
    )

    D = transform_dataset(D, T, None)

    X = concat_features(D)
    print(f'Train size: {X["train"].shape}, Val size {X["val"].shape}')

    if params is None:
        catboost_config = get_catboost_config(data_path, is_cv=True)
    else:
        catboost_config = params

    if 'cat_features' not in catboost_config:
        catboost_config['cat_features'] = list(range(D.n_num_features, D.n_features))

    for col in range(D.n_features):
        for split in X.keys():
            if col in catboost_config['cat_features']:
                X[split][col] = X[split][col].astype(str)
            else:
                X[split][col] = X[split][col].astype(float)
    
    print(T_dict)
    pprint(catboost_config, width=100)
    print('-'*100)
    
    if D.is_regression:
        model = CatBoostRegressor(
            **catboost_config,
            eval_metric='RMSE',
            random_seed=seed,
            max_ctr_complexity = 4
        )
        predict = model.predict
    else:
        model = CatBoostClassifier(
            loss_function="MultiClass" if D.is_multiclass else "Logloss",
            **catboost_config,
            eval_metric='TotalF1',
            random_seed=seed,
            max_ctr_complexity = 4, 
            class_names=[str(i) for i in range(D.n_classes)] if D.is_multiclass else ["0", "1"]
        )
        predict = (
            model.predict_proba
            if D.is_multiclass
            else lambda x: model.predict_proba(x)[:, 1]
        )

    model.fit(
        X['train'], D.y['train'],
        eval_set=(X['val'], D.y['val']),
        verbose=100
    )
    predictions = {k: predict(v) for k, v in X.items()}
    print(predictions['train'].shape)

    report = {}
    report['eval_type'] = eval_type
    report['dataset'] = data_path
    report['metrics'] = D.calculate_metrics(predictions,  None if D.is_regression else 'probs')

    metrics_report = MetricsReport(report['metrics'], D.task_type)
    metrics_report.print_metrics()

    if parent_dir is not None:
        dump_json(report, os.path.join(parent_dir, "results_catboost.json" if model_step=='finetune' else 'pretrain_results_catboost.json'))

    return metrics_report