import os 
import numpy as np
import pandas as pd
import argparse
import math
import json
import time
from typing import Union
from util.util import * 
from preprocess_common.load_data_common import data_preporcesser_common
from util.rho_cdp import cdp_rho

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
parser.add_argument("--marg_num", type=int, default=0)
parser.add_argument("--resample", action='store_true', default = False)
parser.add_argument("--adaptive", action='store_true', default = False)


# hyperparams for debug
parser.add_argument("--test", action="store_true")
parser.add_argument("--lr", type=float, default = None)
parser.add_argument("--iter", type=int, default = None)
parser.add_argument("--batch", type=int, default = None)
parser.add_argument("--graph_sample", action='store_true', default = False)
parser.add_argument("--pub", action='store_true', default = False)
args = parser.parse_args()

if args.sample_device is None:
    args.sample_device = args.device

if args.method in ['rapp', 'rap_syn'] and args.dataset in ['loan', 'higgs-small']:
    print('memory saving mode for JAX')
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
    os.environ["JAX_TRACEBACK_FILTERING"] = "off"

def main(args):
    print(f'privacy setting: ({args.epsilon}, {args.delta})')
    parent_dir, data_path = make_ablation_dir(args)
    time_record = {}

    # data preprocess
    total_rho = cdp_rho(args.epsilon, args.delta)
    data_preprocesser = data_preporcesser_common(args)
    df, domain, preprocesser_divide  = data_preprocesser.load_data(data_path, total_rho) 
    df_pub = data_preprocesser.load_pub_data(data_path)
    if args.method in ['ddpm', 'pe_ddpm', 'ctgan']:
        param_dict = {'rho_used': preprocesser_divide*total_rho} 
    else:
        param_dict = {}

    # fitting model
    start_time = time.time()
    generator_dict = algo_ablation_method(args)(
        args, df=df, domain=domain, 
        rho=(1-preprocesser_divide)*total_rho, 
        parent_dir=parent_dir, 
        preprocesser = data_preprocesser, 
        df_pub=df_pub,
        **param_dict
    )
    end_time = time.time()
    time_record['model fitting time'] = end_time-start_time

    with open(os.path.join(parent_dir, 'time.json'), 'w') as file:
        json.dump(time_record, file)



if __name__ == "__main__":
    main(args)