import os 
import numpy as np
import pandas as pd
import argparse
import pickle
import math
import json
import time
import signal
import sys

from copy import deepcopy
from functools import partial
from typing import Union
from util.util import * 
from preprocess_common.load_data_common import data_preporcesser_common
from util.rho_cdp import cdp_rho
from evaluator.eval_seeds import eval_seeds
from evaluator.eval_sample import eval_sampler

description = ""
formatter = argparse.ArgumentDefaultsHelpFormatter
parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
parser.add_argument("method", help="synthesis method")
parser.add_argument("dataset", help="dataset to use")
parser.add_argument("device", help="device to use")
parser.add_argument("epsilon", type=float, help="privacy parameter")
parser.add_argument("--delta", type=float, default=1e-5, help="privacy parameter")
parser.add_argument("--num_preprocess", type=str, default='privtree')
parser.add_argument("--rare_threshold", type=float, default=0.002) # if 0 then 3sigma
parser.add_argument("--sample_device", help="device to synthesis, only used in some deep learning models", default=None)
parser.add_argument("--resample", action='store_true', default = False)
parser.add_argument("--graph_sample", action='store_true', default = False)
parser.add_argument("--pub", action='store_true', default = False)

# hyperparams for debug
parser.add_argument("--test", action="store_true")
parser.add_argument("--lr", type=float, default = None)
parser.add_argument("--iter", type=int, default = None)
parser.add_argument("--batch", type=int, default = None)
args = parser.parse_args()

if args.sample_device is None:
    args.sample_device = args.device

if args.method in ['rapp', 'rap_syn'] and args.dataset in ['loan', 'gauss50']:
    print('memory saving mode for JAX')
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
    os.environ["JAX_TRACEBACK_FILTERING"] = "off"

# def handler(signum, frame, parent_dir):
#     print("Timeout!!!!!!!!!")
#     warning_dict = {'timeout': 1}
#     with open(os.path.join(parent_dir, 'timeout.json'), 'w') as file:
#         json.dump(warning_dict, file)
#     sys.exit(1)

def main(args):
    print(f'privacy setting: ({args.epsilon}, {args.delta})')
    parent_dir, data_path = make_exp_dir(args)
    time_record = {}

    # data preprocess
    total_rho = cdp_rho(args.epsilon, args.delta)
    print('zCDP rho:', total_rho)
    
    data_preprocesser = data_preporcesser_common(args)
    df, domain, preprocesser_divide  = data_preprocesser.load_data(data_path, total_rho) 
    df_pub = data_preprocesser.load_pub_data(data_path)
    with open(os.path.join(parent_dir, 'preprocesser.pkl'), "wb") as file:
        pickle.dump(data_preprocesser, file)
        
    if args.method in ['ddpm', 'ctgan']:
        param_dict = {'rho_used': preprocesser_divide*total_rho} 
    else:
        param_dict = {}

    # fitting model
    start_time = time.time()
    generator_dict = algo_method(args)(
        args, df=df, domain=domain, 
        rho=(1-preprocesser_divide)*total_rho, 
        parent_dir=parent_dir, 
        preprocesser = data_preprocesser, 
        df_pub=df_pub,
        **param_dict
    )
    end_time = time.time()
    time_record['model fitting time'] = end_time-start_time

    # evaluation
    eval_config = prepare_eval_config(args, parent_dir)
    
    eval_seeds(
        eval_config, 
        sampling_method = args.method,
        device = args.sample_device,
        preprocesser = data_preprocesser,
        time_record = time_record,
        test_data = 'real',
        **generator_dict
    ) 


if __name__ == "__main__":
    main(args)

