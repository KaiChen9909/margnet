import sys
target_path="./"
sys.path.append(target_path)

import os
import numpy as np
import pandas as pd
import torch
import tomli
import itertools
import math
from tqdm import tqdm

from method.MargDL.data.dataset import Dataset
from method.privsyn.PrivSyn.privsyn import PrivSyn
from method.MargDL.scripts.diffusion_model import *
from method.MargDL.scripts.gan import *


def exponential_mechanism(score, rho, sensitivity):
    max_score = np.max(score)
    scaled_score = [s - max_score for s in score]
    exp_score = [np.exp(np.sqrt(2*rho)/sensitivity * s) for s in scaled_score]
    sample_prob = [score/sum(exp_score) for score in exp_score]
    id = np.random.choice(np.arange(len(exp_score)), p = sample_prob)
    return id


class MargDLGen():
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
            df_pub: pd.DataFrame = None
        ):
        self.device = device
        self.dataset = Dataset(df, domain, device)
        self.pub_dataset = Dataset(df_pub, domain, device) if df_pub is not None else None
        self.domain = domain
        self.args = args
        self.parent_dir = parent_dir
        self.model_type = model_type
        self.sample_type = sample_type
        self.save_loss = False

        config['model_params']['data_dim'] = sum(domain.values())
        self.config = config
        if args.iter is not None:
            self.config['train']['selection_iterations'] = args.iter
        if args.lr is not None:
            self.config['train']['lr'] = args.lr
        if args.batch is not None:
            self.config['train']['batch_size'] = args.batch
            
        if args.test:
            self.save_loss = True

        self.reset_model()

    def reset_model(self):
        if self.model_type == 'diffusion':
            self.model = QueryDiffusionG(
                config = self.config,
                domain = self.domain,
                parent_dir = self.parent_dir,
                device = self.device, 
                resample = self.args.resample if self.args.resample else False
            ).to(self.device) 
        elif self.model_type == 'gan':
            self.model = MargGAN(
                config = self.config,
                domain = self.domain,
                parent_dir = self.parent_dir,
                device = self.device, 
                resample = self.args.resample if self.args.resample else False
            )


    @ torch.no_grad()
    def exponential_marginal_selection(self, marginal_candidates, rho_select, rho_measure, candidates_weight):
        score = []
        weight = [candidates_weight[marg] for marg in marginal_candidates]

        syn_marginals = self.model.obtain_sample_marginals(marginal_candidates)
        real_marginals = [self.dataset.marginal_query(marg) for marg in marginal_candidates]

        for i in range(len(marginal_candidates)):
            score.append(weight[i] * (self.dataset.num_records * np.linalg.norm(syn_marginals[i] - real_marginals[i], 1)\
                         - np.sqrt(1/(np.pi * rho_measure)) * real_marginals[i].size))
        
        idx = exponential_mechanism(score, rho_select, max(weight))

        return idx
    

    def fit_adaptive(self, rho):
        select_rho = 0.1*rho/(16*self.dataset.df.shape[1])
        measure_rho = 0.9*rho/(16*self.dataset.df.shape[1])
        rho_used = 0.0
        weight = 1.0
        enhance_weight = self.dataset.df.shape[1]
        # enhance_weight = 1.0

        self.model.initialize_logits()

        one_way_marginals = list(itertools.combinations(self.domain.keys(), 1))
        selected_marginals = [
            (one_way_marginals[i], self.dataset.marginal_query(one_way_marginals[i], measure_rho), weight)
            for i in range(len(one_way_marginals))
        ]
        rho_used += measure_rho * len(one_way_marginals)

        print('-'*100)
        print('Initialization')
        self.model.store_marginals(selected_marginals, 1.0)
        self.model.train_model(
            self.config['train']['lr'], 
            self.config['train']['selection_iterations'],
            save_loss = self.save_loss,
            path_prefix = 'init'
        )
        print('-'*100)

        two_way_marginals = list(itertools.combinations(self.domain.keys(), 2))
        # marg_candidates = one_way_marginals + two_way_marginals
        # candidates_select_weight = {marg: 1.0 for marg in one_way_marginals} | {marg: 2.0 for marg in two_way_marginals}
        candidates_mask = {marg: 1 for marg in one_way_marginals}
        terminate = False

        round = 1
        while not terminate:
            marg_candidates = two_way_marginals

            candidates_select_weight = {marg: 2.0 for marg in marg_candidates}

            id = self.exponential_marginal_selection(marg_candidates, select_rho, measure_rho, candidates_select_weight)
            marg = marg_candidates[id]

            if marg not in candidates_mask.keys():
                candidates_mask[marg] = 1
            else:
                candidates_mask[marg] += 1
            print('selected marginal:', marg)

            # enhance_weight = np.sqrt(np.prod(self.domain[attr] for attr in marg))
            one_selected_marginals = [(marg, self.dataset.marginal_query(marg, measure_rho), enhance_weight*weight)]
            selected_marginals += one_selected_marginals
            w_t = self.model.obtain_sample_marginals([marg])[0]

            # self.model.reset_model()
            self.model.store_marginals(selected_marginals, enhance_weight)   
            self.model.train_model(
                self.config['train']['lr'], 
                self.config['train']['selection_iterations'],
                save_loss = self.save_loss,
                path_prefix = f'{round}'
            )
            selected_marginals[-1] = (one_selected_marginals[0][0], one_selected_marginals[0][1], weight)

            rho_used += measure_rho + select_rho
            if rho_used + measure_rho + select_rho > rho:
                weight = weight * np.sqrt(0.9 * (rho - rho_used)/measure_rho)
                measure_rho = 0.9*(rho - rho_used) 
                select_rho = 0.1*(rho - rho_used) 
                terminate = True
            else:
                w_t_plus_1 = self.model.obtain_sample_marginals([marg])[0]
                if self.dataset.num_records * np.linalg.norm(w_t_plus_1 - w_t, 1) < np.sqrt(1/(measure_rho * np.pi)) * w_t_plus_1.size:
                    if candidates_mask[marg] == 1:
                        print('-'*100)
                        print('!!!!!!!!!!!!!!!!! sigma updated')
                        weight *= np.sqrt(2)
                        measure_rho *= 2
                        select_rho *= 2
            
            print('-'*100)
            round += 1

        print('finish marginal selection')
        print('selected marginals:', list(candidates_mask.keys()))
        self.model.store_marginals(selected_marginals, 1.0)
        self.model.train_model(
                self.config['train']['lr'], 
                self.config['train']['final_iterations'],
                save_loss = self.save_loss,
                path_prefix = 'final'
            )
        
        return selected_marginals

    def sample(self, num_samples, preprocesser=None, parent_dir=None):
        syn_data = self.model.sample(num_samples)

        preprocesser.reverse_data(syn_data, parent_dir)
        return syn_data 
   


def MargDLGen_ablation(MargDLGen):
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
        assert (marg_num > 0), 'Must a positive number of marginals'
        rng = np.random.default_rng(seed)

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
        self.report_l1_loss(
            report_marginals,
            self.parent_dir,
            'raw'
        )




def marggan_main(args, df, domain, rho, **kwargs):
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

    generator.fit_adaptive(rho)

    return {'MargDL_generator': generator}


def margdiff_main(args, df, domain, rho, **kwargs):
    config_path = os.path.join('method/MargDL/config/diffusion', f'{args.dataset}.toml')
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
        model_type = 'diffusion',
        sample_type = 'graphical' if args.graph_sample else 'direct'
    )

    generator.fit_adaptive(rho)

    return {'MargDL_generator': generator}



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
