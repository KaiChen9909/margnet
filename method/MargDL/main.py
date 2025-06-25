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
    mean_score = np.max(score)
    exp_score = [np.exp(np.sqrt(2*rho)/sensitivity * (s - mean_score)) for s in score]
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


    def fit_fast(self, rho):
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

        init_two_way_marginals = None

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


# def margdiff_main(args, df, domain, rho, **kwargs):
#     config = read_config(args)

#     generator = MargDLGen(
#         args = args,
#         df = df,
#         domain = domain,
#         device = args.device,
#         config = config,
#         parent_dir = kwargs.get('parent_dir', None),
#         model_type='diffusion'
#     )

#     generator.fit_adaptive(rho)

#     return {'MargDL_generator': generator}



    # def fit_spe(self, rho):
    #     self.model.initialize_logits()
    #     weight = 1.0

    #     one_way_marginals = list(itertools.combinations(self.domain.keys(), 1))
    #     selected_marginals = [
    #         (one_way_marginals[i], self.dataset.marginal_query(one_way_marginals[i], 0.1*rho/len(one_way_marginals)), 1.0) # use 0.1rho for initialization
    #         for i in range(len(one_way_marginals))
    #     ]

    #     np.set_printoptions(formatter={'float': '{:.4f}'.format})
    #     print(self.dataset.marginal_query(('cat_attr_3', )))
    #     print(self.dataset.marginal_query(('cat_attr_4', )))

    #     self.model.store_marginals(selected_marginals)
    #     self.model.train_model('train', self.config['train']['lr'], self.config['train']['warmup_iterations'], use_target=True)

    #     marg = ('cat_attr_3', 'cat_attr_4')
    #     selected_marginals.append(
    #         (marg, self.dataset.marginal_query(marg, 0.9*rho), 1.0)
    #     )

    #     syn_marg = self.model.obtain_sample_marginals([('cat_attr_3', 'cat_attr_4')], num_samples=10240)[0]
    #     real_marg = selected_marginals[-1][1]

    #     np.set_printoptions(formatter={'float': '{:.4f}'.format})
    #     print(real_marg.reshape((self.domain['cat_attr_3'], self.domain['cat_attr_4'])))
    #     print('error:', np.sum(np.abs(syn_marg - real_marg)))

    #     # self.reset_model()
    #     self.model.store_marginals(selected_marginals)
    #     self.model.train_model('train', self.config['train']['lr'], self.config['train']['warmup_iterations'], use_target=True)

    #     syn_marg = self.model.obtain_sample_marginals([('cat_attr_3', 'cat_attr_4')], num_samples=10240)[0]
    #     real_marg = selected_marginals[-1][1]
        
    #     np.set_printoptions(formatter={'float': '{:.4f}'.format})
    #     print(syn_marg.reshape((self.domain['cat_attr_3'], self.domain['cat_attr_4'])))
    #     print('error:', np.sum(np.abs(syn_marg - real_marg)))

    #     raise ValueError('debug')


    # def fit_mst(self, rho):
    #     self.model.initialize_logits()

    #     one_way_marginals = list(itertools.combinations(self.domain.keys(), 1))
    #     rho_rate = self.rho_allocation(one_way_marginals)
    #     one_way_selected_marginals = [
    #         (one_way_marginals[i], self.dataset.marginal_query(one_way_marginals[i], 0.1*rho*rho_rate[i]), 1.0) # use 0.1rho for initialization
    #         for i in range(len(one_way_marginals))
    #     ]

    #     if self.config['train']['warmup_iterations'] > 0:
    #         self.model.store_marginals(one_way_selected_marginals)
    #         self.model.train_model('train', self.config['train']['lr'], self.config['train']['warmup_iterations'], use_target=True) #warm up training
        
    #     two_way_marginals = self.mst_select(one_way_selected_marginals, 0.1*rho, 0.1*rho*rho_rate)
    #     rho_rate = self.rho_allocation(two_way_marginals)
    #     two_way_selected_marginals = [
    #         (two_way_marginals[i], self.dataset.marginal_query(two_way_marginals[i], 0.8*rho*rho_rate[i]), 1.0) # use 0.1rho for initialization
    #         for i in range(len(two_way_marginals))
    #     ]
    #     selected_marginals = one_way_selected_marginals + two_way_selected_marginals
    #     # selected_marginals = two_way_selected_marginals
    #     self.model.store_marginals(selected_marginals)   
    #     self.model.train_model('train', self.config['train']['lr'], self.config['train']['iterations'], use_target=True)
        
    # def complete_marginals(self, one_way, two_way):
    #     all_marginals = list(itertools.combinations(self.domain.keys(), 2))
    #     measurements = []

    #     for (marg1, marg2) in all_marginals:
    #         if (marg1, marg2) in two_way:
    #             measurements.append(((marg1, marg2), *two_way[(marg1, marg2)]))
    #         else:
    #             matrix = np.outer(one_way[(marg1,)][0], one_way[(marg2,)][0]).flatten()
    #             measurements.append(((marg1, marg2), matrix, 1.0))

    #     return measurements

    # def fit_complete(self, rho):
    #     one_way_marginals = list(itertools.combinations(self.domain.keys(), 1))
    #     rho_rate = self.rho_allocation(one_way_marginals)
    #     one_way_selected_marginals = {
    #         one_way_marginals[i]: (self.dataset.marginal_query(one_way_marginals[i], 0.1*rho*rho_rate[i]), 1.0) # use 0.1rho for initialization
    #         for i in range(len(one_way_marginals))
    #     }

    #     two_way_marginals = self.mst_select([(k, *v) for (k,v) in one_way_selected_marginals.items()], 0.1*rho, 0.1*rho*rho_rate)
    #     rho_rate = self.rho_allocation(two_way_marginals)
    #     two_way_selected_marginals = {
    #         two_way_marginals[i]: (self.dataset.marginal_query(two_way_marginals[i], 0.8*rho*rho_rate[i]), 1.0) # use 0.1rho for initialization
    #         for i in range(len(two_way_marginals))
    #     }

    #     selected_marginals = self.complete_marginals(one_way_selected_marginals, two_way_selected_marginals)
    #     self.model.store_marginals(selected_marginals)   
    #     self.model.train_model('train', self.config['train']['lr'], self.config['train']['iterations'], use_target=True)

    # def fit_test_indep(self):
    #     self.model.initialize_logits()

    #     one_way_marginals = list(itertools.combinations(self.domain.keys(), 1))
    #     one_way_selected_marginals = [
    #         (one_way_marginals[i], self.dataset.marginal_query(one_way_marginals[i]), 1.0)
    #         for i in range(len(one_way_marginals))
    #     ]

    #     if self.config['train']['warmup_iterations'] > 0:
    #         self.model.store_marginals(one_way_selected_marginals)
    #         self.model.train_model('train', self.config['train']['lr'], self.config['train']['warmup_iterations'], use_target=True) #warm up training
        
    #     print('------------------ one-way test --------------------')
    #     syn_marg = self.model.obtain_sample_marginals([('cat_attr_1',), ('y_attr', )], num_samples=10240)
    #     real_marg = [self.dataset.marginal_query(('cat_attr_1', )), self.dataset.marginal_query(('y_attr', ))]

    #     print('1 error_syn_real', np.sum(np.abs(syn_marg[0] - real_marg[0])))        
    #     print('1 error_syn_real', np.sum(np.abs(syn_marg[1] - real_marg[1])))  

    #     two_way_marginals = [('num_attr_1', 'num_attr_2'), ('num_attr_1', 'cat_attr_1'), ('num_attr_2', 'cat_attr_1')]
    #     two_way_selected_marginals = [
    #         (two_way_marginals[i], self.dataset.marginal_query(two_way_marginals[i]), 1.0)
    #         for i in range(len(two_way_marginals))
    #     ]
        
    #     selected_marginals = one_way_selected_marginals + two_way_selected_marginals
    #     self.model.store_marginals(selected_marginals)   
    #     self.model.train_model('train', self.config['train']['lr'], self.config['train']['iterations'], use_target=True)

    #     print('------------------ two-way test --------------------')

    #     syn_marg = self.model.obtain_sample_marginals([('cat_attr_1', 'y_attr')], num_samples=10240)[0]
    #     real_marg = self.dataset.marginal_query(('cat_attr_1', 'y_attr'))
    #     print(syn_marg)
    #     print(real_marg)

    #     syn_indep_marg_list = self.model.obtain_sample_marginals([('cat_attr_1',), ('y_attr',)], num_samples=10240)
    #     syn_indep_marg = np.einsum('i,j->ij', syn_indep_marg_list[0], syn_indep_marg_list[1]).flatten()
    #     indep_marg = np.einsum('i,j->ij', self.dataset.marginal_query(('cat_attr_1', )), self.dataset.marginal_query(('y_attr', ))).flatten()
    #     print(indep_marg)

    #     print('2 error_syn_real', np.sum(np.abs(syn_marg - real_marg)))
    #     print('2 error_syn_synindep', np.sum(np.abs(syn_marg - syn_indep_marg)))
    #     print('2 error_syn_indep', np.sum(np.abs(syn_marg - indep_marg)))
    #     print('2 error_real_indep', np.sum(np.abs(real_marg - indep_marg)))


    #     print('------------------ one-way test --------------------')
    #     syn_marg = self.model.obtain_sample_marginals([('cat_attr_1',), ('y_attr', )], num_samples=10240)
    #     real_marg = [self.dataset.marginal_query(('cat_attr_1', )), self.dataset.marginal_query(('y_attr', ))]

    #     print('1 error_syn_real', np.sum(np.abs(syn_marg[0] - real_marg[0])))        
    #     print('1 error_syn_real', np.sum(np.abs(syn_marg[1] - real_marg[1])))  

    #     raise ValueError('test end')




# class test_model():
#     def __init__(
#             self, 
#             df: pd.DataFrame, 
#             domain: dict, 
#             device: str, 
#             config: dict,
#             **kwargs
#         ):
#         self.device = device
#         self.dataset = Dataset(df, domain, device)
#         self.column_dims = domain
#         self.config = config

#         self.domain = GEMDomain(domain.keys(), domain.values())
#         self.data = GEMDataset(df, self.domain)
        
#         self.num_classes = np.array(list(domain.values()))
#         self.column_name = np.array(list(domain.keys()))
#         self.cum_num_classes = np.concatenate((np.array([0]), np.cumsum(self.num_classes)))

#         config['denosier_params']['d_in'] = sum(domain.values())
#         config['denosier_params']['dim_t'] = sum(domain.values())

#         self.model = Generator(
#             embedding_dim =  config['denosier_params']['d_in'],
#             gen_dims =  (480, 480),
#             data_dim = config['denosier_params']['d_in']
#         ).to(self.device)
#         self.lr = 1e-3

#         mean = torch.zeros(self.config['train']['batch_size'], self.config['denosier_params']['d_in']).to(self.device)
#         std = mean+1
#         self.input = torch.normal(mean=mean, std=std)

#         # marginals_dict = {}
#         # logits_list = []

#         # for col_name, col_size in zip(self.column_name, self.num_classes):
#         #     if col_name in marginals_dict:
#         #         arr = marginals_dict[col_name]
#         #         if len(arr) != col_size:
#         #             raise ValueError('Invalid one-way marginal')
#         #     else:
#         #         arr = np.ones(col_size, dtype=np.float32) / col_size

#         #     logits_list.append(torch.tensor(arr, device=self.device))

#         # logits = torch.cat(logits_list, dim=0)
#         # z_oh = []
#         # for i in range(len(self.cum_num_classes)-1):
#         #     start = self.cum_num_classes[i]
#         #     end = self.cum_num_classes[i+1]

#         #     probs = logits[start: end]
#         #     idxs = torch.multinomial(probs, self.config['train']['batch_size'], replacement=True)
#         #     z_oh.append(F.one_hot(idxs, num_classes=(end - start)).float())

#         # self.input = torch.cat(z_oh, dim=1)
    
#     def find_query_index(self, marginals):
#         index = []
#         answer = []
#         size = []
#         for (marg, matrix, _) in marginals:
#             start_idx = [self.cum_num_classes[np.where(self.column_name == col)[0][0]] for col in marg]
#             end_idx = [start + self.column_dims[col] for start, col in zip(start_idx, marg)]
#             iter_list = [range(a,b) for (a, b) in zip(start_idx, end_idx)]
#             index += list(itertools.product(*iter_list))
#             answer += matrix.flatten().tolist()
#         # return torch.tensor(index, device=self.device), torch.tensor(answer, device=self.device)
#         return torch.tensor(index).to(self.device).long(), torch.tensor(answer, dtype=torch.float64, device=self.device)

#     def store_marginals_gem(self, marginals, rho=None):
#         N = self.dataset.num_records

#         query_manager = QueryManager(self.data.domain, marginals)
#         real_answers = np.concatenate(query_manager.get_answer(self.data, concat=False))
#         self.real_answers = torch.tensor(real_answers).to(self.device)
        
#         if rho:
#             np.random.seed(42)
#             noise = np.random.normal(loc=0, scale=(1/N)*np.sqrt(len(marginals)/(2 * rho)))
#             # noise = torch.tensor(np.random.normal(loc=0, scale=(1/N)*np.sqrt(len(marginals)/(2 * rho)), size=self.real_answers.shape)).to(self.device)
#             self.real_answers += noise 
#             self.real_answers = torch.clamp(self.real_answers, 0, 1)

#         self.queries = torch.tensor(query_manager.queries).to(self.device).long()

#     def store_marginals(self, marginals, rho=None):
#         self.marginal_list = [marg_list[0] for marg_list in marginals]
#         self.queries, self.real_answers = self.find_query_index(marginals)

#     def activate(self, x):
#         data = []
#         for i in range(len(self.cum_num_classes)-1):
#             st = self.cum_num_classes[i]
#             ed = self.cum_num_classes[i+1]
#             data.append(x[:, st:ed].softmax(-1))
#         return torch.cat(data, dim=1)

#     def run_train_loop(self, info, lr, iterations):
#         self.optimizer = torch.optim.Adam(
#             self.model.parameters(), 
#             lr=lr
#         )
#         self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, iterations, eta_min=1e-8)
#         for iter in range(iterations):
#             self.optimizer.zero_grad()
#             output = self.activate(self.model(self.input))

#             fake_query_attr = [output[:, q] for q in self.queries]
#             fake_answer = [attr.prod(-1).mean(axis=0) for attr in fake_query_attr]
#             real_answer = [measurement.clone() for measurement in self.real_answers]

#             loss = sum((real_answer[i] - fake_answer[i]).abs() for i in range(len(real_answer)))
#             print(loss)
#             loss.backward()
#             self.optimizer.step()
#             self.scheduler.step()

#     def train(self, rho):
#         one_way_marginals = list(itertools.combinations(self.dataset.domain.keys(), 1))

#         selected_marginals = [
#             (marg, self.dataset.marginal_query(marg, 0.1*rho/len(one_way_marginals)), np.sqrt(0.1 * rho / len(one_way_marginals))) # use 0.1rho for initialization
#             for marg in one_way_marginals
#         ]

#         self.store_marginals_gem(one_way_marginals, 0.1*rho)
#         # self.store_marginals(selected_marginals, 0.1*rho)
#         self.run_train_loop('initialization', self.config['train']['lr'], 10)

#         marg_list = PrivSyn.two_way_marginal_selection(self.dataset.df, self.dataset.domain, 0.1*rho, 0.8*rho)

#         selected_marginals = [
#             (marg, self.dataset.marginal_query(marg, 0.8*rho/len(marg_list)), 1.0) #np.sqrt(0.8*rho/rounds)
#             for marg in marg_list
#         ]

#         self.store_marginals_gem(marg_list, 0.8*rho)
#         # self.store_marginals(selected_marginals, 0.8*rho)
#         self.run_train_loop('train', self.config['train']['lr'], self.config['train']['iterations'])
    

#     def test(self, rho):
#         marg_list = [('cat_attr_8', 'cat_attr_9')]
#         # marg_list = [('cat_attr_9', 'cat_attr_11'), ('cat_attr_11', 'y_attr'), ('cat_attr_8', 'cat_attr_11'), ('cat_attr_7', 'cat_attr_11'), 
#         #              ('cat_attr_5', 'cat_attr_11'), ('cat_attr_1', 'cat_attr_11'), ('cat_attr_10', 'cat_attr_11'), ('cat_attr_3', 'cat_attr_4'), 
#         #              ('cat_attr_9', 'cat_attr_12'), ('cat_attr_12', 'y_attr'), ('cat_attr_8', 'cat_attr_10'), ('cat_attr_9', 'cat_attr_10'), 
#         #              ('cat_attr_10', 'y_attr'), ('cat_attr_1', 'cat_attr_10'), ('cat_attr_5', 'cat_attr_10')]
#         selected_marginals = [
#             (marg, self.dataset.marginal_query(marg), 1) for marg in marg_list
#         ]

#         self.store_marginals(selected_marginals)
#         ans1 = self.real_answers
#         id1 = self.queries
#         print(self.real_answers)
#         print(self.queries)

#         self.store_marginals_gem(marg_list)
#         ans2 = self.real_answers
#         id2 = self.queries
#         print(self.real_answers)
#         print(self.queries)

#         print(torch.equal(ans1, ans2))
#         print(torch.equal(id1, id2))
    
#     def train_simple(self, rho):
#         one_way_marginals = [('cat_attr_7', 'cat_attr_9')]
#         selected_marginals = [
#             (marg, self.dataset.marginal_query(marg, 0.1*rho/len(one_way_marginals)), np.sqrt(0.1 * rho / len(one_way_marginals))) # use 0.1rho for initialization
#             for marg in one_way_marginals
#         ]

#         self.store_marginals(selected_marginals)

#         self.model.train()
#         self.run_train_loop('initialization', self.config['train']['lr'], 10)
#         print(self.queries)
#         print(len(self.queries))
#         raise 'debug'
    
#     @ torch.no_grad()
#     def sample(self, num_samples, preprocesser=None, parent_dir=None):
#         n_batch = int(np.ceil(num_samples)/self.config['train']['batch_size'])
#         ordinal_datas = []
#         for _ in range(n_batch):
#             logits = self.activate(self.model(self.input))
#             cum_num_classes = torch.tensor([0] + list(torch.cumsum(torch.tensor(self.num_classes, dtype=torch.int64), dim=0)))

#             ordinal_tensors = [
#                 torch.multinomial(logits[:, cum_num_classes[i]:cum_num_classes[i+1]], num_samples=logits.shape[0], replacement=True)
#                 for i in range(len(self.num_classes))
#             ]
#             ordinal_datas.append(torch.stack(ordinal_tensors, dim=1))
            
#         ordinal_datas = torch.concat(ordinal_datas, dim=0).cpu().numpy()
#         if preprocesser is not None:
#             preprocesser.reverse_data(ordinal_datas, parent_dir)
#         return ordinal_datas



'''
    def mst_select(self, marginals, rho, rho_measure):
        from method.AIM.mbi.MST import MST_select
        from method.AIM.mbi.matrix import Identity

        one_way_measurement = []
        for i in range(len(marginals)):
            marg, measure, _ = marginals[i]
            I = Identity(measure.size)
            one_way_measurement.append((I, measure, np.sqrt(1/(2*rho_measure[i])), marg))
        
        return MST_select(self.dataset.df, self.domain, rho, one_way_measurement)
    def fit(self, rho):
        self.model.initialize_logits()

        one_way_marginals = list(itertools.combinations(self.domain.keys(), 1))
        rho_rate = self.rho_allocation(one_way_marginals)
        one_way_selected_marginals = [
            (one_way_marginals[i], self.dataset.marginal_query(one_way_marginals[i], 0.1*rho*rho_rate[i]), 1.0) # use 0.1rho for initialization
            for i in range(len(one_way_marginals))
        ]

        if self.config['train']['warmup_iterations'] > 0:
            self.model.store_marginals(one_way_selected_marginals)
            self.model.train_model('train', self.config['train']['lr'], self.config['train']['warmup_iterations'], use_target=True) #warm up training

        two_way_marginals = PrivSyn.two_way_marginal_selection(self.dataset.df, self.dataset.domain, 0.1*rho, 0.8*rho)
        rho_rate = self.rho_allocation(two_way_marginals)
        two_way_selected_marginals = [
            (two_way_marginals[i], self.dataset.marginal_query(two_way_marginals[i], 0.8*rho*rho_rate[i]), 1.0) # use 0.1rho for initialization
            for i in range(len(two_way_marginals))
        ]
        selected_marginals = one_way_selected_marginals + two_way_selected_marginals
        # selected_marginals = two_way_selected_marginals
        self.model.store_marginals(selected_marginals)   
        self.model.train_model('train', self.config['train']['lr'], self.config['train']['iterations'], use_target=True)


    def fit_fix(self, rho):
        one_way_marginals = list(itertools.combinations(self.domain.keys(), 1))
        selected_marginals = [
            (one_way_marginals[i], self.dataset.marginal_query(one_way_marginals[i], 0.1*rho/len(one_way_marginals)), 1)
            for i in range(len(one_way_marginals))
        ]
        self.model.store_marginals(selected_marginals)
        self.model.train_model('train', self.config['train']['lr'], self.config['train']['warmup_iterations'], use_target=True)

        two_way_marginals = {('cat_attr_5', 'cat_attr_7'): 1, ('cat_attr_7', 'cat_attr_9'): 2, ('cat_attr_7', 'y_attr'): 2, ('cat_attr_3', 'cat_attr_4'): 1, ('cat_attr_6', 'cat_attr_9'): 1, ('cat_attr_4', 'y_attr'): 1, ('cat_attr_6', 'y_attr'): 1, ('cat_attr_10', 'y_attr'): 1, ('cat_attr_8', 'y_attr'): 1, ('cat_attr_1', 'y_attr'): 1, ('num_attr_1', 'y_attr'): 1, ('cat_attr_1', 'cat_attr_6'): 1, ('num_attr_2', 'y_attr'): 1, ('cat_attr_3', 'y_attr'): 1, ('cat_attr_1', 'cat_attr_9'): 1, ('cat_attr_8', 'cat_attr_9'): 1, ('cat_attr_5', 'cat_attr_9'): 1, ('num_attr_1', 'cat_attr_5'): 1}
        sum_w = sum(two_way_marginals.values())
        selected_marginals += [
            (k, self.dataset.marginal_query(k, 0.9*rho/sum_w), v)
            for k,v in two_way_marginals.items()
        ]
        self.model.store_marginals(selected_marginals)
        self.model.train_model('train', self.config['train']['lr'], self.config['train']['warmup_iterations'], use_target=True)

        syn = self.model.obtain_sample_marginals([marg[0] for marg in selected_marginals])
        real = [marg[1] for marg in selected_marginals]
        err = [np.sum(np.abs(syn[i] - real[i])) for i in range(len(selected_marginals))]
        print([f'{x:.4f}' for x in err])


    def fit_test(self, rho):
        two_way_marginals = list(itertools.combinations(self.domain.keys(), 2))
        selected_marginals = [
            (two_way_marginals[i], self.dataset.marginal_query(two_way_marginals[i], rho/len(two_way_marginals)), 1.0)
            for i in range(len(two_way_marginals))
        ]

        self.model.store_marginals(selected_marginals)   
        self.model.train_model('train', self.config['train']['lr'], self.config['train']['warmup_iterations'], use_target=True)


    def fit_test_adap(self, rho):
        two_way_marginals = list(itertools.combinations(self.domain.keys(), 2))
        
        # selected_marginals = [
        #     (two_way_marginals[i], self.dataset.marginal_query(two_way_marginals[i], rho/len(two_way_marginals)), 1.0)
        #     for i in range(len(two_way_marginals))
        # ]
        # self.model.store_marginals(selected_marginals)   
        # self.model.train_model_treelike('train', self.config['train']['lr'], self.config['train']['warmup_iterations'], use_target=True)

        selected_marginals = []
        for marg in two_way_marginals:
            selected_marginals.append(
                (marg, self.dataset.marginal_query(marg, rho/len(two_way_marginals)), 1.0)
            )
            self.model.store_marginals(selected_marginals)   
            self.model.train_model('train', self.config['train']['lr'], self.config['train']['warmup_iterations'], use_target=True)

    def fit_hard(self, rho):
        marg = ('cat_attr_3', 'cat_attr_4')
        selected_marginals = []
        selected_marginals.append(
                (marg, self.dataset.marginal_query(marg, rho), 1.0)
            )
        
        self.model.store_marginals(selected_marginals)   
        self.model.train_model('train', self.config['train']['lr'], self.config['train']['warmup_iterations'], use_target=True)

        syn = self.model.obtain_sample_marginals([marg])[0]
        real = selected_marginals[0][1]
        print('50 error:', np.sum(np.abs(syn - real)))

        self.model.train_model('train', self.config['train']['lr'], self.config['train']['warmup_iterations'], use_target=True)

        syn = self.model.obtain_sample_marginals([marg])[0]
        real = selected_marginals[0][1]
        print('100 error:', np.sum(np.abs(syn - real)))

        self.model.train_model('train', self.config['train']['lr'], self.config['train']['warmup_iterations'], use_target=True)

        syn = self.model.obtain_sample_marginals([marg])[0]
        real = selected_marginals[0][1]
        print('150 error:', np.sum(np.abs(syn - real)))

        self.model.train_model('train', self.config['train']['lr'], self.config['train']['warmup_iterations'], use_target=True)

        syn = self.model.obtain_sample_marginals([marg])[0]
        real = selected_marginals[0][1]
        print('200 error:', np.sum(np.abs(syn - real)))



        for t in reversed(range(steps)):
            self.optimizer = torch.optim.Adam(
                self.model._denoise_fn.parameters(), 
                lr=lr
            )
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, iterations, eta_min=1e-8)
            loss_list = []
            for iter in range(iterations):
                self.optimizer.zero_grad()
                loss = self.model.compute_loss(t)
                loss.backward()

                loss_list.append(loss.item())
                if iter == 0:
                    start_loss = loss.item()
                elif iter == iterations-1:
                    end_loss = loss.item()
                print(loss.item())

                self.optimizer.step()
                self.scheduler.step()
            
            print(f'step {t+1} optimization: {start_loss:.4f} -> {end_loss:.4f}')
            self.model.update_logits(t)
            self.model.save_model(t)
            # self.model.reset_generator()
            track[f'step {t+1}'] = loss_list
'''