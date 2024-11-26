#######################################################################

# This file will use Privsyn + PGM and Gumbel adaptive + PGM

#######################################################################

import os 
import sys
target_path="./"
sys.path.append(target_path)
import numpy as np
import pandas as pd
import argparse
import itertools
import json
from tqdm import tqdm
from AIM.src.mbi import (
    Dataset,
    Domain,
    estimation,
    junction_tree,
    LinearMeasurement,
    LinearMeasurement,
)
from AIM.mechanisms.mechanism import Mechanism
from collections import defaultdict
# from AIM.src.mbi.matrix import Identity
from privsyn.PrivSyn.privsyn import PrivSyn

def powerset(iterable):
    "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(1, len(s) + 1)
    )


def downward_closure(Ws):
    ans = set()
    for proj in Ws:
        ans.update(powerset(proj))
    return list(sorted(ans, key=len))


# def hypothetical_model_size(domain, cliques):
#     model = GraphicalModel(domain, cliques)
#     return model.size * 8 / 2 ** 20
def hypothetical_model_size(domain, cliques):
    jtree, _ = junction_tree.make_junction_tree(domain, cliques)
    maximal_cliques = junction_tree.maximal_cliques(jtree)
    cells = sum(domain.size(cl) for cl in maximal_cliques)
    size_mb = cells * 8 / 2**20
    return size_mb


def compile_workload(workload):
    # this is different from aim's work, since other selection methods allocate same budget to all marginals
    weights = {cl: wt for (cl, wt) in workload}
    workload_cliques = weights.keys()

    return {cl: 1 for cl in downward_closure(workload_cliques)}


def filter_candidates(candidates, model, size_limit):
    ans = {}
    free_cliques = downward_closure(model.cliques)
    for cl in candidates:
        cond1 = (
            hypothetical_model_size(model.domain, model.cliques + [cl]) <= size_limit
        )
        cond2 = cl in free_cliques
        if cond1 or cond2:
            ans[cl] = candidates[cl]
    return ans


class Gumbel_Generator(Mechanism):
    def __init__(
        self,
        epsilon=1.0,
        delta=1e-5,
        rho=None,
        bounded=None,
        # rounds=None,
        max_model_size=80,
        max_iters=1000,
        structural_zeros={},
    ):
        if rho is None:
            super().__init__(epsilon, delta, bounded)
        else:
            self.rho = rho 
            self.prng = np.random
            self.bouned = bounded
        # self.rounds = rounds
        self.max_iters = max_iters
        self.max_model_size = max_model_size
        self.structural_zeros = structural_zeros

    def gumbel_mechanism(self, k, qualities, rho, base_measure=None):
        if isinstance(qualities, dict):
            # import pandas as pd
            # print(pd.Series(list(qualities.values()), list(qualities.keys())).sort_values().tail())
            keys = list(qualities.keys())
            qualities = np.array([qualities[key] for key in keys])
            if base_measure is not None:
                base_measure = np.log([base_measure[key] for key in keys])
        else:
            qualities = np.array(qualities)
            keys = np.arange(qualities.size)

        """ Sample a candidate from the permute-and-flip mechanism """
        noisy_qualities = qualities + np.random.gumbel(loc=0.0, scale=k/(len(qualities) * np.sqrt(2*rho)), size=len(qualities))

        if len(noisy_qualities) >= k:
            selected_idx = np.argpartition(noisy_qualities, -k)[-k:]
            return [keys[i] for i in selected_idx]
        else:
            return keys
    
    def worst_approximated_gumbel(self, candidates, answers, model, k, rho):
        errors = {}
        for cl in candidates:
            x = answers[cl]
            xest = model.project(cl).datavector()
            errors[cl] = np.linalg.norm(x - xest, 1)

        return self.gumbel_mechanism(k, errors, rho)


    def run_gumbel(self, data, k, workload):
        # rounds = self.rounds or 16 * len(data.domain)
        T = int(2 * np.ceil(len(data.domain)/k))
        candidates = compile_workload(workload) # all possible subset of two-way marginals
        answers = {cl: data.project(cl).datavector() for cl in candidates}

        print(f"Total rounds: {T}, number of marginal candidates: {len(candidates)}")

        measurements = []
        marginal_dict = {}

        initial_cliques = [
                cl for cl in candidates if len(cl) == 1
            ]
        for cl in initial_cliques: 
            n = data.domain.size(cl)
            x = data.project(cl).datavector()
            sigma = np.sqrt(len(initial_cliques)/(0.2*self.rho)) # use 0.1rho
            y = x + self.gaussian_noise(sigma, x.size) # uniformly random initialize the model
            measurements.append(LinearMeasurement(y, cl, stddev=sigma))
            if cl not in marginal_dict.keys():
                marginal_dict[cl] = 1
            else:
                marginal_dict[cl] += 1

        self.model = estimation.mirror_descent(
            data.domain, measurements, iters=self.max_iters,
        )

        for t in tqdm(range(T)):
            size_limit = self.max_model_size * (t+1) / T
            small_candidates = filter_candidates(candidates, self.model, size_limit)
            sigma = np.sqrt(k*T/(0.9 *self.rho)) # use 0.45rho

            cl_set = self.worst_approximated_gumbel(
                small_candidates, answers, self.model, k, (0.45*self.rho)/T  
            ) # use 0.45rho

            for cl in cl_set: 
                n = data.domain.size(cl)
                x = data.project(cl).datavector()
                y = x + self.gaussian_noise(sigma, x.size)
                measurements.append(LinearMeasurement(y, cl, stddev=sigma))
                # z = self.model.project(cl).datavector()

                if cl not in marginal_dict.keys():
                    marginal_dict[cl] = 1
                else:
                    marginal_dict[cl] += 1

                self.model = estimation.mirror_descent(
                    data.domain, measurements, iters=self.max_iters, potentials=potentials
                )
                # w = self.model.project(cl).datavector()

        print("Start model construction")
        self.model = estimation.mirror_descent(
            data.domain, measurements, iters=self.max_iters
        )
        print("Finish model construction")

        return self.model, marginal_dict

    def syn_data(
            self, 
            num_synth_rows, 
            path = None,
            preprocesser = None
        ):
        synth = self.model.synthetic_data(rows=num_synth_rows)
        if path is None:
            print('This is the raw data needed to be decoded')
            return synth
        else:
            synth.save_data_npy(path, preprocesser)
            return None


class Privsyn_Generator(Mechanism):
    def __init__(
        self,
        epsilon=1.0,
        delta=1e-5,
        rho=None,
        bounded=None,
        # rounds=None,
        max_model_size=80,
        max_iters=1000,
        structural_zeros={},
    ):
        if rho is None:
            super().__init__(epsilon, delta, bounded)
        else:
            self.rho = rho 
            self.prng = np.random
            self.bouned = bounded
        # self.rounds = rounds
        self.max_iters = max_iters
        self.max_model_size = max_model_size
        self.structural_zeros = structural_zeros
    
    def worst_approximated_gumbel(self, candidates, answers, model, k, rho):
        errors = {}
        for cl in candidates:
            x = answers[cl]
            xest = model.project(cl).datavector()
            errors[cl] = np.linalg.norm(x - xest, 1)

        return self.gumbel_mechanism(k, errors, rho)


    def run_privsyn(self, data, k, workload):
        # rounds = self.rounds or 16 * len(data.domain)
        candidates = compile_workload(workload) # all possible subset of two-way marginals
        answers = {cl: data.project(cl).datavector() for cl in candidates}

        measurements = []
        marginal_dict = {}

        initial_cliques = [
                cl for cl in candidates if len(cl) == 1
            ]
        for cl in initial_cliques:
            n = data.domain.size(cl)
            x = data.project(cl).datavector()
            sigma = np.sqrt(len(initial_cliques)/(0.2*self.rho)) # use 0.1rho
            y = x + self.gaussian_noise(sigma, n) # uniformly random initialize the model
            measurements.append(LinearMeasurement(y, cl, stddev=sigma))
            if cl not in marginal_dict.keys():
                marginal_dict[cl] = 1
            else:
                marginal_dict[cl] += 1

        zeros = self.structural_zeros
        self.model = estimation.mirror_descent(
            data.domain, measurements, iters=self.max_iters,
        )

        cl_set = PrivSyn.two_way_marginal_selection(data.df, data.domain.config, 0.1*self.rho, 0.8*self.rho) # use 0.1rho 
        print("Selected marginals:", cl_set)
        
        sigma = np.sqrt(len(cl_set)/(1.6*self.rho)) # use 0.8rho 
        for cl in cl_set: 
            n = data.domain.size(cl)
            x = data.project(cl).datavector()
            y = x + self.gaussian_noise(sigma, n)
            measurements.append(LinearMeasurement(y, cl, stddev=sigma))

        print("Start model construction")
        self.model = estimation.mirror_descent(
            data.domain, measurements, iters=self.max_iters,
        )
        print("Finish model construction")
        
        return self.model, cl_set

    def syn_data(
            self, 
            num_synth_rows, 
            path = None,
            preprocesser = None
        ):
        synth = self.model.synthetic_data(rows=num_synth_rows)
        if path is None:
            print('This is the raw data needed to be decoded')
            return synth
        else:
            synth.save_data_npy(path, preprocesser)
            return None



def add_default_params(args):
    args.max_model_size = 100
    args.max_iters = 1000
    args.degree = 2
    args.num_marginals = None 
    args.max_cells = 100000
    args.k = 5
    return args


def gumbel_select_main(args, df, domain, rho, **kwargs):
    args = add_default_params(args)
    domain = Domain(domain.keys(), domain.values())
    data = Dataset(df, domain)

    workload = list(itertools.combinations(data.domain, args.degree))
    workload = [cl for cl in workload if data.domain.size(cl) <= args.max_cells]
    if args.num_marginals is not None:
        workload = [
            workload[i]
            for i in np.random.choice(len(workload), args.num_marginals, replace=False)
        ]

    workload = [(cl, 1.0) for cl in workload]
    mech = Gumbel_Generator(
        rho = rho,
        max_model_size=args.max_model_size,
        max_iters=args.max_iters,
    )
    mech.run_gumbel(data, args.k, workload)

    return {'gumbel_select_generator': mech} 


def privsyn_select_main(args, df, domain, rho, **kwargs):
    args = add_default_params(args)
    domain = Domain(domain.keys(), domain.values())
    data = Dataset(df, domain)

    workload = list(itertools.combinations(data.domain, args.degree))
    workload = [cl for cl in workload if data.domain.size(cl) <= args.max_cells]
    if args.num_marginals is not None:
        workload = [
            workload[i]
            for i in np.random.choice(len(workload), args.num_marginals, replace=False)
        ]

    workload = [(cl, 1.0) for cl in workload]
    mech = Privsyn_Generator(
        rho = rho,
        max_model_size=args.max_model_size,
        max_iters=args.max_iters,
    )
    mech.run_privsyn(data, args.k, workload)

    return {'privsyn_select_generator': mech}