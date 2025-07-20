import sys
target_path="./"
sys.path.append(target_path)

import os
import numpy as np
import pandas as pd
import torch
import tomli
import json
import itertools
import math
from tqdm import tqdm

from method.MargDL.data.dataset import Dataset
from method.privsyn.PrivSyn.privsyn import PrivSyn
from method.AIM.mst import MST
from method.MargDL.scripts.diffusion_model import *
from method.MargDL.scripts.gan import *
from method.MargDL.main import MargDLGen


def exponential_mechanism(score, rho, sensitivity):
    max_score = np.max(score)
    scaled_score = [s - max_score for s in score]
    exp_score = [np.exp(np.sqrt(2*rho)/sensitivity * s) for s in scaled_score]
    sample_prob = [score/sum(exp_score) for score in exp_score]
    id = np.random.choice(np.arange(len(exp_score)), p = sample_prob)
    return id

class MargDLGen_ablation(MargDLGen):
    def __init__(
            self, 
            args,
            df: pd.DataFrame,  
            domain: dict, 
            device: str, 
            config: dict, 
            parent_dir: str,
            model_type: str,
            sample_type: str = 'direct',
            df_pub: pd.DataFrame = None):
        super().__init__(args, df, domain, device, config, parent_dir, model_type, sample_type, df_pub)

    def report_detailed_error(self, selected_marginals):
        all_margs = list(itertools.combinations(self.domain.keys(), 2))
        sel_margs = [marg[0] for marg in selected_marginals]
        sel_label = [
            1 if marg in sel_margs else 0 
            for marg in all_margs
        ]

        syn = self.model.obtain_sample_marginals(all_margs)
        real = [self.dataset.marginal_query(marg) for marg in all_margs]

        errors = [np.sum(np.abs(syn[i]-real[i])) for i in range(len(all_margs))]

        error_df = pd.DataFrame({
            'label': sel_label,
            'error': errors
        })
        error_df.to_csv(os.path.join(self.parent_dir, 'marg error.csv'))



    def report_l1_loss(self, selected_marginals, parent_dir, report_type = 'stat'):
        import json 

        report = {'marg_num': len(selected_marginals)}
        loss = []
        for marg, real_res, _ in selected_marginals:
            syn_res = self.model.obtain_sample_marginals([marg])[0]
            loss.append(
                np.sum(np.abs(syn_res - real_res))
            )
        
        if report_type == 'stat':
            report['mean loss'] = np.mean(loss)
            report['max loss'] = np.max(loss)
            report['min loss'] = np.min(loss)
            report['sum loss'] = np.sum(loss)
            report['loss std'] = np.std(loss)
        else:
            report['loss'] = loss

        with open(os.path.join(parent_dir, 'fitting error.json'), 'w') as file:
            json.dump(report, file)


    def fit_random_adap(self, marg_num, save_loss=False, seed=109):
        assert (marg_num > 0), 'Must a positive number of marginals'
        rng = np.random.default_rng(seed)
        self.model.initialize_logits()

        two_way_marginals = list(itertools.combinations(self.domain.keys(), 2))
        marginal_range = np.arange(len(two_way_marginals))
        marginal_index = rng.choice(marginal_range, size=marg_num, replace=False)

        selected_marginals = []
        for i in marginal_index:
            selected_marginals.append(
                (two_way_marginals[i], self.dataset.marginal_query(two_way_marginals[i], None), 1.0)
            )
            self.model.store_marginals(selected_marginals, 1.0)
            self.model.train_model(
                self.config['train']['lr'], 
                self.config['train']['selection_iterations']
            )
        
        self.report_l1_loss(
            selected_marginals,
            self.parent_dir
        )

    def fit_random(self, marg_num, save_loss=False, seed=109):
        torch.cuda.reset_peak_memory_stats()
        assert (marg_num > 0), 'Must a positive number of marginals'
        rng = np.random.default_rng(seed)
        self.model.initialize_logits()

        two_way_marginals = list(itertools.combinations(self.domain.keys(), 2))
        marginal_range = np.arange(len(two_way_marginals))
        marginal_index = rng.choice(marginal_range, size=marg_num, replace=False)

        selected_marginals = []
        for i in marginal_index:
            selected_marginals.append(
                (two_way_marginals[i], self.dataset.marginal_query(two_way_marginals[i], None), 1.0)
            )
            self.model.store_marginals(selected_marginals, 1.0)

        self.model.train_model(
            self.config['train']['lr'], 
            self.config['train']['selection_iterations'],
            save_loss
        )
        self.report_l1_loss(
            selected_marginals,
            self.parent_dir
        )

    def fit_all(self, save_loss=False):
        two_way_marginals = list(itertools.combinations(self.domain.keys(), 2))
        selected_marginals = []
        self.model.initialize_logits()

        for i in range(len(two_way_marginals)):
            selected_marginals.append(
                (two_way_marginals[i], self.dataset.marginal_query(two_way_marginals[i], None), 1.0)
            )
            self.model.store_marginals(selected_marginals, 1.0)

        self.model.train_model(
            self.config['train']['lr'], 
            2000,
            save_loss
        )
        
        attrs = list(self.domain.keys())
        report_marginal_list = [(attrs[0], attrs[i + 1]) for i in range(len(attrs) - 1)]
        report_marginals = []
        for i in range(len(report_marginal_list)):
            report_marginals.append(
                (report_marginal_list[i], self.dataset.marginal_query(report_marginal_list[i], None), 1.0)
            )
        peak = torch.cuda.max_memory_allocated(device=torch.device(self.device))
        memory_dict = {'memory (MB)': peak/1024**2}
        with open(os.path.join(self.parent_dir, 'memory.json'), 'w') as file:
            json.dump(memory_dict, file)

        self.report_l1_loss(
            report_marginals,
            self.parent_dir,
            'raw'
        )
    

    def fit_fix(self, rho, **kwargs):
        # import json
        # with open('exp/adult/aim/1.0_privtree_0.002/marginal.json', 'rb') as file:
        #     file_marginals = json.load(file)
        
        self.model.initialize_logits()

        # selected_marginals = []
        # for k,v in file_marginals.items():
        #     marg = tuple(k.split(','))
        #     for rho in v:
        #         selected_marginals = [
        #             (marg, self.dataset.marginal_query(marg, rho), np.sqrt(rho))
        #         ]
        one_way_marginals = list(itertools.combinations(self.domain.keys(), 1))
        selected_marginals = [
            (one_way_marginals[i], self.dataset.marginal_query(one_way_marginals[i], 0.1*rho/len(one_way_marginals)), 1.0)
            for i in range(len(one_way_marginals))
        ]

        marginal_list = PrivSyn.two_way_marginal_selection(self.dataset.df, self.domain, 0.1*rho, 0.8*rho)
        selected_marginals += [
            (marginal_list[i], self.dataset.marginal_query(marginal_list[i], 0.8*rho/len(marginal_list)), 1.0)
            for i in range(len(marginal_list))
        ]
        
        self.model.store_marginals(selected_marginals, 1.0)
        self.model.train_model(
            self.config['train']['lr'], 
            2000
        )

        return 0



def marggan_ablation_main(args, df, domain, rho, **kwargs):
    config_path = os.path.join('method/MargDL/config/gan', f'{args.dataset}.toml')
    with open(config_path, 'rb') as file:
        config = tomli.load(file)[f'{args.epsilon}']

    generator = MargDLGen_ablation(
        args = args,
        df = df,
        df_pub = kwargs.get('df_pub', None),
        domain = domain,
        device = args.device,
        config = config,
        parent_dir = kwargs.get('parent_dir', None),
        model_type = 'gan',
        sample_type = 'graphical' if args.graph_sample else 'direct'
    )

    if not args.adaptive:
        config['train']['selection_iterations'] = max(config['train']['selection_iterations']*args.marg_num, 1000)
        generator.fit_random(args.marg_num, save_loss=True)
    else:
        generator.fit_random_adap(args.marg_num, save_loss=True)

    return {'MargDL_generator': generator}



def marggan_ablation_all(args, df, domain, rho, **kwargs):
    config_path = os.path.join('method/MargDL/config/gan', f'{args.dataset}.toml')
    with open(config_path, 'rb') as file:
        config = tomli.load(file)[f'{args.epsilon}']

    generator = MargDLGen_ablation(
        args = args,
        df = df,
        df_pub = kwargs.get('df_pub', None),
        domain = domain,
        device = args.device,
        config = config,
        parent_dir = kwargs.get('parent_dir', None),
        model_type = 'gan',
        sample_type = 'graphical' if args.graph_sample else 'direct'
    )

    generator.fit_all(save_loss=True)

    return {'MargDL_generator': generator}




def marggan_ablation_marg(args, df, domain, rho, **kwargs):
    config_path = os.path.join('method/MargDL/config/gan', f'{args.dataset}.toml')
    with open(config_path, 'rb') as file:
        config = tomli.load(file)[f'{args.epsilon}']

    generator = MargDLGen(
        args = args,
        df = df,
        df_pub = kwargs.get('df_pub', None),
        domain = domain,
        device = args.device,
        config = config,
        parent_dir = kwargs.get('parent_dir', None),
        model_type = 'gan',
        sample_type = 'graphical' if args.graph_sample else 'direct'
    )

    selected_marginals = generator.fit_adaptive(rho)
    generator.report_detailed_error(selected_marginals)

    return {'MargDL_generator': generator}


def marggan_temp(args, df, domain, rho, **kwargs):
    config_path = os.path.join('method/MargDL/config/gan', f'{args.dataset}.toml')
    with open(config_path, 'rb') as file:
        config = tomli.load(file)[f'{args.epsilon}']

    generator = MargDLGen_ablation(
        args = args,
        df = df,
        df_pub = kwargs.get('df_pub', None),
        domain = domain,
        device = args.device,
        config = config,
        parent_dir = kwargs.get('parent_dir', None),
        model_type = 'gan',
        sample_type = 'graphical' if args.graph_sample else 'direct'
    )

    generator.fit_fix(rho)

    return {'MargDL_generator': generator}


def margdiff_temp(args, df, domain, rho, **kwargs):
    config_path = os.path.join('method/MargDL/config/diffusion', f'{args.dataset}.toml')
    with open(config_path, 'rb') as file:
        config = tomli.load(file)[f'{args.epsilon}']

    generator = MargDLGen_ablation(
        args = args,
        df = df,
        df_pub = kwargs.get('df_pub', None),
        domain = domain,
        device = args.device,
        config = config,
        parent_dir = kwargs.get('parent_dir', None),
        model_type = 'diffusion',
        sample_type = 'graphical' if args.graph_sample else 'direct'
    )

    selected_marginals = generator.fit_fix(rho)

    return {'MargDL_generator': generator}