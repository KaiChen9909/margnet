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
from method.MargDL.scripts.diffusion_model import *
from method.MargDL.scripts.gan import *


def exponential_mechanism(score, rho, sensitivity):
    max_score = np.max(score)
    scaled_score = [s - max_score for s in score]
    exp_score = [np.exp(np.sqrt(2*rho)/sensitivity * s) for s in scaled_score]
    sample_prob = [score/sum(exp_score) for score in exp_score]
    id = np.random.choice(np.arange(len(exp_score)), p = sample_prob)
    return id


def obtain_indep_marginals(one_selected_marginals, two_way_marginals):
    marginal_map = {
        cols[0]: (marg, w)
        for cols, marg, w in one_selected_marginals
    }

    for cl in two_way_marginals:
        marg_list = [marginal_map[attr][0] for attr in cl]
        w_list = [marginal_map[attr][1] for attr in cl]

        joint = reduce(np.multiply.outer, marg_list).flatten()
        joint_weight = max(w_list)

        one_selected_marginals.append((cl, joint, joint_weight))
    
    return one_selected_marginals



class MargDLGen_ab():
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
            raise NotImplementedError
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
        real_marginals = [self.dataset.marginal_query(marg, scale=False) for marg in marginal_candidates]

        for i in range(len(marginal_candidates)):
            score.append(weight[i] * (np.linalg.norm(self.dataset.est_num_records * syn_marginals[i] - real_marginals[i], 1)\
                         - np.sqrt(1/(np.pi * rho_measure)) * real_marginals[i].size)) 
        
        idx = exponential_mechanism(score, rho_select, max(weight))

        return idx
    
    def exponential_query_selection(self, known_total, rho_select, **kwargs):
        marginal_candidates = list(itertools.combinations(self.domain.keys(), 2))
        query_candidates = self.dataset.obtain_all_query_index()

        real_ans = self.dataset.obtain_all_query(scale=False)
        syn_marg = self.model.obtain_sample_marginals(marginal_candidates)
        syn_ans = []
        for marg in syn_marg:
            syn_ans += list(known_total * marg.flatten())
        
        score = [np.abs(real_ans[i] - syn_ans[i]) for i in range(len(real_ans))]
        idx = exponential_mechanism(score, rho_select, 1.0)
        return query_candidates[idx]
        
    
    def fit_update(self, rho):
        torch.cuda.reset_peak_memory_stats()

        select_rho = 0.1*rho/(16*self.dataset.df.shape[1])
        measure_rho = 0.9*rho/(16*self.dataset.df.shape[1])
        rho_used = 0.0
        weight = 1.0
        enhance_weight = self.dataset.df.shape[1]
        # enhance_weight = 1.0

        one_way_marginals = list(itertools.combinations(self.domain.keys(), 1))
        selected_marginals = [
            (one_way_marginals[i], self.dataset.marginal_query(one_way_marginals[i], measure_rho, update_records=True), weight)
            for i in range(len(one_way_marginals))
        ]
        rho_used += measure_rho * len(one_way_marginals)
    
        print('-'*100)
        print('Initialization')
        self.model.store_marginals(selected_marginals)
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
            self.model.store_marginals(selected_marginals)   
            self.model.train_model(
                self.config['train']['lr'], 
                self.config['train']['selection_iterations'],
                save_loss = self.save_loss,
                path_prefix = f'{round}'
            )
            selected_marginals[-1] = (one_selected_marginals[0][0], one_selected_marginals[0][1], weight)
            rho_used += measure_rho + select_rho

            # privacy budget allocation update
            # no limitation on selected time
            w_t_plus_1 = self.model.obtain_sample_marginals([marg])[0]
            if self.dataset.est_num_records * np.linalg.norm(w_t_plus_1 - w_t, 1) < np.sqrt(1/(measure_rho * np.pi)) * w_t_plus_1.size:
                print('-'*100)
                print('!!!!!!!!!!!!!!!!! sigma updated')
                weight *= np.sqrt(2)
                measure_rho *= 2
                select_rho *= 2

            # termination condition
            if rho_used + measure_rho + select_rho > rho:
                weight = weight * np.sqrt(0.9 * (rho - rho_used)/measure_rho)
                measure_rho = 0.9*(rho - rho_used) 
                select_rho = 0.1*(rho - rho_used) 
                terminate = True
            
            print('-'*100)
            round += 1

        print('finish marginal selection')
        print('selected marginals:', list(candidates_mask.keys()))
        self.model.store_marginals(selected_marginals)
        self.model.train_model(
                self.config['train']['lr'], 
                self.config['train']['final_iterations'],
                save_loss = self.save_loss,
                path_prefix = 'final'
            )
        
        peak = torch.cuda.max_memory_allocated(device=torch.device(self.device))
        memory_dict = {'memory (MB)': peak/1024**2}
        with open(os.path.join(self.parent_dir, 'memory.json'), 'w') as file:
            json.dump(memory_dict, file)

        return selected_marginals
    

    def fit_adaptive_weight(self, rho):
        torch.cuda.reset_peak_memory_stats()

        select_rho = 0.1*rho/(16*self.dataset.df.shape[1])
        measure_rho = 0.9*rho/(16*self.dataset.df.shape[1])
        rho_used = 0.0
        weight = 1.0
        enhance_weight = 1.0

        one_way_marginals = list(itertools.combinations(self.domain.keys(), 1))
        selected_marginals = [
            (one_way_marginals[i], self.dataset.marginal_query(one_way_marginals[i], measure_rho, update_records=True), 1.0)
            for i in range(len(one_way_marginals))
        ]
        rho_used += measure_rho * len(one_way_marginals)
    
        print('-'*100)
        print('Initialization')
        self.model.store_marginals(selected_marginals)
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
            one_selected_marginals = [(marg, self.dataset.marginal_query(marg, measure_rho), 1.0)]
            selected_marginals += one_selected_marginals
            w_t = self.model.obtain_sample_marginals([marg])[0]

            # self.model.reset_model()
            self.model.store_marginals(selected_marginals)   
            self.model.train_model(
                self.config['train']['lr'], 
                self.config['train']['selection_iterations'],
                save_loss = self.save_loss,
                path_prefix = f'{round}'
            )
            selected_marginals[-1] = (one_selected_marginals[0][0], one_selected_marginals[0][1], 1.0)
            rho_used += measure_rho + select_rho

            # privacy budget allocation update
            w_t_plus_1 = self.model.obtain_sample_marginals([marg])[0]
            if self.dataset.est_num_records * np.linalg.norm(w_t_plus_1 - w_t, 1) < np.sqrt(1/(measure_rho * np.pi)) * w_t_plus_1.size:
                if candidates_mask[marg] == 1:
                    print('-'*100)
                    print('!!!!!!!!!!!!!!!!! sigma updated')
                    weight *= np.sqrt(2)
                    measure_rho *= 2
                    select_rho *= 2

            # termination condition
            if rho_used + measure_rho + select_rho > rho:
                weight = weight * np.sqrt(0.9 * (rho - rho_used)/measure_rho)
                measure_rho = 0.9*(rho - rho_used) 
                select_rho = 0.1*(rho - rho_used) 
                terminate = True
            
            print('-'*100)
            round += 1

        print('finish marginal selection')
        print('selected marginals:', list(candidates_mask.keys()))
        self.model.store_marginals(selected_marginals)
        self.model.train_model(
                self.config['train']['lr'], 
                self.config['train']['final_iterations'],
                save_loss = self.save_loss,
                path_prefix = 'final'
            )
        
        peak = torch.cuda.max_memory_allocated(device=torch.device(self.device))
        memory_dict = {'memory (MB)': peak/1024**2}
        with open(os.path.join(self.parent_dir, 'memory.json'), 'w') as file:
            json.dump(memory_dict, file)

        return selected_marginals
    

    def fit_adaptive_query(self, rho):
        torch.cuda.reset_peak_memory_stats()

        select_rho = 0.1*rho/(16*self.dataset.df.shape[1])
        measure_rho = 0.9*rho/(16*self.dataset.df.shape[1])
        rho_used = 0.0
        weight = 1.0
        enhance_weight = self.dataset.df.shape[1]

        one_way_queries_ans = self.dataset.obtain_all_query(rho = measure_rho * self.dataset.df.shape[1], order=1, update_records=True)
        one_way_queries = np.arange(len(one_way_queries_ans))
        selected_queries = [
            ((i,), one_way_queries_ans[i], 1.0)
            for i in range(len(one_way_queries))
        ]

        print(f'len query: {len(one_way_queries_ans)}')

        known_total = self.dataset.est_num_records
        rho_used += measure_rho * len(one_way_queries)

    
        print('-'*100)
        print('Initialization')
        self.model.store_queries(selected_queries)
        self.model.train_model(
            self.config['train']['lr'], 
            self.config['train']['selection_iterations'],
            save_loss = self.save_loss,
            path_prefix = 'init'
        )
        print('-'*100)


        candidates_mask = {marg: 1 for marg in one_way_queries}
        terminate = False

        round = 1
        while not terminate:
            q = self.exponential_query_selection(known_total, select_rho)

            if q not in candidates_mask.keys():
                candidates_mask[q] = 1
            else:
                candidates_mask[q] += 1
            print('selected marginal:', q)

            # enhance_weight = np.sqrt(np.prod(self.domain[attr] for attr in marg))
            one_selected_query = [(q, self.dataset.query_from_indices(q, measure_rho), 1.0*enhance_weight)]
            selected_queries += one_selected_query
            w_t = self.model.obtain_sample_queries([q])[0]

            # self.model.reset_model()
            self.model.store_queries(selected_queries)   
            self.model.train_model(
                self.config['train']['lr'], 
                self.config['train']['selection_iterations'],
                save_loss = self.save_loss,
                path_prefix = f'{round}'
            )
            selected_queries[-1] = (one_selected_query[0][0], one_selected_query[0][1], 1.0)
            rho_used += measure_rho + select_rho

            # privacy budget allocation update
            w_t_plus_1 = self.model.obtain_sample_queries([q])[0]
            if self.dataset.est_num_records * np.abs(w_t_plus_1 - w_t) < np.sqrt(1/(measure_rho * np.pi)):
                if candidates_mask[q] == 1:
                    print('-'*100)
                    print('!!!!!!!!!!!!!!!!! sigma updated')
                    weight *= np.sqrt(2)
                    measure_rho *= 2
                    select_rho *= 2

            # termination condition
            if rho_used + measure_rho + select_rho > rho:
                weight = weight * np.sqrt(0.9 * (rho - rho_used)/measure_rho)
                measure_rho = 0.9*(rho - rho_used) 
                select_rho = 0.1*(rho - rho_used) 
                terminate = True
            
            print('-'*100)
            round += 1

        print('finish marginal selection')
        print('selected marginals:', list(candidates_mask.keys()))
        self.model.store_queries(selected_queries)
        self.model.train_model(
                self.config['train']['lr'], 
                self.config['train']['final_iterations'],
                save_loss = self.save_loss,
                path_prefix = 'final'
            )
        
        peak = torch.cuda.max_memory_allocated(device=torch.device(self.device))
        memory_dict = {'memory (MB)': peak/1024**2}
        with open(os.path.join(self.parent_dir, 'memory.json'), 'w') as file:
            json.dump(memory_dict, file)

        return selected_queries
    

    def fit_iter(self, rho):
        torch.cuda.reset_peak_memory_stats()

        init_rho = 0.1*rho/self.dataset.df.shape[1]
        select_rho = 0.45*rho/(5*self.dataset.df.shape[1])
        measure_rho = 0.45*rho/(5*self.dataset.df.shape[1])

        enhance_weight = self.dataset.df.shape[1]

        one_way_marginals = list(itertools.combinations(self.domain.keys(), 1))
        selected_marginals = [
            (one_way_marginals[i], self.dataset.marginal_query(one_way_marginals[i], init_rho, update_records=True), 1.0)
            for i in range(len(one_way_marginals))
        ]
    
        print('-'*100)
        print('Initialization')
        self.model.store_marginals(selected_marginals)
        self.model.train_model(
            self.config['train']['lr'], 
            self.config['train']['selection_iterations'],
            save_loss = self.save_loss,
            path_prefix = 'init'
        )
        print('-'*100)

        two_way_marginals = list(itertools.combinations(self.domain.keys(), 2))
        candidates_mask = {marg: 1 for marg in one_way_marginals}

        for round in range(1, 5*self.dataset.df.shape[1] + 1):
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
            one_selected_marginals = [(marg, self.dataset.marginal_query(marg, measure_rho), enhance_weight)]
            selected_marginals += one_selected_marginals

            # self.model.reset_model()
            self.model.store_marginals(selected_marginals)   
            self.model.train_model(
                self.config['train']['lr'], 
                self.config['train']['selection_iterations'],
                save_loss = self.save_loss,
                path_prefix = f'{round}'
            )
            selected_marginals[-1] = (one_selected_marginals[0][0], one_selected_marginals[0][1], 1.0) # reset the weight
            print('-'*100)

        print('finish marginal selection')
        print('selected marginals:', list(candidates_mask.keys()))
        self.model.store_marginals(selected_marginals)
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
   


def marggan_ablation_update(args, df, domain, rho, **kwargs):
    config_path = os.path.join('method/MargDL/config/gan', f'{args.dataset}.toml')
    with open(config_path, 'rb') as file:
        config = tomli.load(file)[f'{args.epsilon}']

    generator = MargDLGen_ab(
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

    generator.fit_update(rho)
    # generator.fit_adaptive_indep_back(rho)

    return {'MargDL_generator': generator}



def marggan_ablation_noweight(args, df, domain, rho, **kwargs):
    config_path = os.path.join('method/MargDL/config/gan', f'{args.dataset}.toml')
    with open(config_path, 'rb') as file:
        config = tomli.load(file)[f'{args.epsilon}']

    generator = MargDLGen_ab(
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

    generator.fit_adaptive_weight(rho)
    # generator.fit_adaptive_indep_back(rho)

    return {'MargDL_generator': generator}


def marggan_ablation_query(args, df, domain, rho, **kwargs):
    config_path = os.path.join('method/MargDL/config/gan', f'{args.dataset}.toml')
    with open(config_path, 'rb') as file:
        config = tomli.load(file)[f'{args.epsilon}']

    generator = MargDLGen_ab(
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

    generator.fit_adaptive_query(rho)
    # generator.fit_adaptive_indep_back(rho)

    return {'MargDL_generator': generator}


def marggan_ablation_iter(args, df, domain, rho, **kwargs):
    config_path = os.path.join('method/MargDL/config/gan', f'{args.dataset}.toml')
    with open(config_path, 'rb') as file:
        config = tomli.load(file)[f'{args.epsilon}']

    generator = MargDLGen_ab(
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

    generator.fit_iter(rho)
    # generator.fit_adaptive_indep_back(rho)

    return {'MargDL_generator': generator}