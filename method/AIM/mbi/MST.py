import numpy as np
from method.AIM.mbi.Dataset import Dataset
from method.AIM.mbi.Domain import Domain
from method.AIM.mbi.inference import FactoredInference
from scipy import sparse
from scipy.cluster.hierarchy import DisjointSet
import networkx as nx
import itertools
from method.AIM.cdp2adp import cdp_rho
from scipy.special import logsumexp
import argparse

def exponential_mechanism(q, eps, sensitivity, prng=np.random, monotonic=False):
    coef = 1.0 if monotonic else 0.5
    scores = coef * eps / sensitivity * q
    probas = np.exp(scores - logsumexp(scores))
    return prng.choice(q.size, p=probas)


def MST_select(df, domain, rho, measurement, cliques=[]):
    '''
    This is a simplification of MST selection, the input is
    df: a dataframe
    domain: a dictionary of attribute domain
    rho: privacy budget for selection 
    measurement: a list of measurement of one-way marginals, must be in form [I, clique, measurement, sigma]
    '''
    domain = Domain(domain.keys(), domain.values())
    data = Dataset(df, domain)

    engine = FactoredInference(
            data.domain, iters=1000, warm_start=True, structural_zeros={}
        )
    est = engine.estimate(measurement)

    weights = {}
    candidates = list(itertools.combinations(data.domain.attrs, 2))
    for a, b in candidates:
        xhat = est.project([a, b]).datavector()
        x = data.project([a, b]).datavector()
        weights[a, b] = np.linalg.norm(x - xhat, 1)

    T = nx.Graph()
    T.add_nodes_from(data.domain.attrs)
    ds = DisjointSet(data.domain.attrs)

    for e in cliques:
        T.add_edge(*e)
        ds.merge(*e)

    r = len(list(nx.connected_components(T)))
    epsilon = np.sqrt(8 * rho / (r - 1))
    for i in range(r - 1):
        candidates = [e for e in candidates if not ds.connected(*e)]
        wgts = np.array([weights[e] for e in candidates])
        idx = exponential_mechanism(wgts, epsilon, sensitivity=1.0)
        e = candidates[idx]
        T.add_edge(*e)
        ds.merge(*e)

    return list(T.edges)