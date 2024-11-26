############################################################################

# This file aims to use privsyn marginal selection and gsd synthesizers
# The marginal selection is from PrivSyn

############################################################################
import sys
target_path="./"
sys.path.append(target_path)
import numpy as np
import argparse
import os 
import json
import math

from private_gsd.utils.utils_data import Dataset, Domain
from private_gsd.stats import Marginals, ChainedStatistics
from private_gsd.models import GSD 
from private_gsd.utils.cdp2adp import cdp_rho
from jax.random import PRNGKey
from privsyn.PrivSyn.privsyn import PrivSyn


def prepare_domain(domain):
    domain_new = {}
    for k,v in domain.items():
        if k.split('_')[0] == 'num':
            domain_new[k] = 1
        else:
            domain_new[k] = v
    return domain_new


def gsd_syn_main(args, df, domain, rho, **kwargs): 
    domain_new = Domain.fromdict(prepare_domain(domain))
    # domain = Domain.fromdict(domain)
    data = Dataset(df, domain_new)

    marginals = PrivSyn.two_way_marginal_selection(data.df, data.domain.config, 0.1*rho, 0.9*rho)
    marginal_module2 = Marginals.get_outside_kway_combinations(marginals, data.domain, k=2, bins=[2, 4, 8, 16, 32])
    stat_module = ChainedStatistics([marginal_module2])
    stat_module.fit(data)

    true_stats = stat_module.get_all_true_statistics()
    stat_fn = stat_module._get_workload_fn()

    algo = GSD(
        domain=data.domain,
        print_progress=True,
        stop_early=True,
        # num_generations=20000,
        population_size_muta=50,
        population_size_cross=50,
        # data_size = df.shape[0]
    ) 

    key = PRNGKey(0)
    algo.zcdp_syn_init(key, stat_module, 0.9*rho)

    return {'gsd_syn_generator': algo}