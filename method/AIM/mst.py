import os 
import sys
target_path="./"
sys.path.append(target_path)
import numpy as np
import pandas as pd
import argparse
import itertools
import json
import networkx as nx
from scipy.optimize import bisect
from scipy.cluster.hierarchy import DisjointSet
from scipy.special import logsumexp
from collections import defaultdict

from method.AIM.mbi.Dataset import Dataset
from method.AIM.mbi.inference import FactoredInference
from method.AIM.mbi.graphical_model import GraphicalModel
from method.AIM.mbi.Domain import Domain
from method.AIM.mbi.Factor import Factor
from method.AIM.mechanism import Mechanism
from method.AIM.mbi.matrix import Identity
from evaluator.eval_seeds import eval_seeds
from method.AIM.cdp2adp import cdp_rho 



def exponential_mechanism(q, eps, sensitivity, prng=np.random, monotonic=False):
    coef = 1.0 if monotonic else 0.5
    scores = coef * eps / sensitivity * q
    probas = np.exp(scores - logsumexp(scores))
    return prng.choice(q.size, p=probas)


class MST(Mechanism):
    def __init__(
        self,
        parent_dir,
        epsilon=1.0,
        delta=1e-5,
        rho=None,
        bounded=None,
        rounds=None,
        max_model_size=80,
        max_iters=1000,
        structural_zeros={},
    ):
        if rho is None:
            super(MST, self).__init__(epsilon, delta, bounded)
        else:
            self.rho = rho 
            self.prng = np.random
            self.bouned = bounded
        self.rounds = rounds
        self.max_iters = max_iters
        self.max_model_size = max_model_size
        self.structural_zeros = structural_zeros
        self.parent_dir = parent_dir
    
    
    def run(self, data, initial_cliques=None):
        initial_cliques = [(attr,) for attr in data.domain.attrs]

        measurements = []
        marginal_dict = {}

        sigma = np.sqrt(3 / (2 * len(initial_cliques) * self.rho))
        for cl in initial_cliques:
            x = data.project(cl).datavector()
            y = x + self.gaussian_noise(sigma, x.size)
            I = Identity(y.size)
            measurements.append((I, y, sigma, cl))

            k = ",".join(map(str, cl)) if isinstance(cl, tuple) else cl
            if k not in marginal_dict.keys():
                marginal_dict[k] = [1/(2 * sigma**2)]
            else:
                marginal_dict[k].append(1/(2 * sigma**2))

        engine = FactoredInference(
            data.domain, iters=self.max_iters, warm_start=True, structural_zeros={}
        )
        est = engine.estimate(measurements)

        weights = {}
        candidates = list(itertools.combinations(data.domain.attrs, 2))
        for a, b in candidates:
            xhat = est.project([a, b]).datavector()
            x = data.project([a, b]).datavector()
            weights[a, b] = np.linalg.norm(x - xhat, 1)

        T = nx.Graph()
        T.add_nodes_from(data.domain.attrs)
        ds = DisjointSet(data.domain.attrs)


        r = len(list(nx.connected_components(T)))
        epsilon = np.sqrt(8 * self.rho / (3*(r - 1)))
        for i in range(r - 1):
            candidates = [e for e in candidates if not ds.connected(*e)]
            wgts = np.array([weights[e] for e in candidates])
            idx = exponential_mechanism(wgts, epsilon, sensitivity=1.0, prng=self.prng)
            e = candidates[idx]
            T.add_edge(*e)
            ds.merge(*e)
        
        two_way_cliques = list(T.edges)
        sigma = np.sqrt(3 / (2 * len(two_way_cliques) * self.rho))
        for cl in two_way_cliques:
            x = data.project(cl).datavector()
            y = x + self.gaussian_noise(sigma, x.size)
            I = Identity(y.size)
            measurements.append((I, y, sigma, cl))

            k = ",".join(map(str, cl)) if isinstance(cl, tuple) else cl
            if k not in marginal_dict.keys():
                marginal_dict[k] = [1/(2 * sigma**2)]
            else:
                marginal_dict[k].append(1/(2 * sigma**2))

        engine = FactoredInference(
            data.domain, iters=self.max_iters, warm_start=True, structural_zeros={}
        )
        self.model = engine.estimate(measurements)
        print("Finish model construction")
        with open(os.path.join(self.parent_dir, 'marginal.json'), 'w') as file:
            json.dump(marginal_dict, file, indent=4)
    
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
    args.max_iters = 1000
    args.num_marginals = None 
    args.max_cells = 250000
    return args 


def mst_main(args, df, domain, rho, **kwargs):
    args = add_default_params(args)
    domain = Domain(domain.keys(), domain.values())
    data = Dataset(df, domain)

    mech = MST(
        rho = rho,
        parent_dir = kwargs.get('parent_dir', None),
        max_iters=args.max_iters,
    )
    mech.run(data)

    return {'aim_generator': mech}
        