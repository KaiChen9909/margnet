################################################################

# This is the main implementation of PrivTree discretizer
# mostly from https://github.com/ruankie/differentially-private-range-queries

################################################################


import numpy as np
import pandas as pd
import warnings

def get_domain_subdomains(domain):
    
    # domain = [left, right]
    # left = domain[0]
    # right = domain[1]
    
    # -----------
    # | q1 | q2 |
    # -----------

    dom_w = domain[1] - domain[0]
    
    q1 = [domain[0], domain[1] - dom_w/2]
    q2 = [domain[0] + dom_w/2, domain[1]]
    
    return q1, q2


def is_in_domain(x, left, right): #includes left and bottom border
        if (x >= left) and (x < right):
            return True
        else:
            return False

def count_in_domain(xs, domain):
    count = 0
    
    for i in range(xs.shape[0]):
        if is_in_domain(xs[i], domain[0], domain[1]):
            count += 1

    return count

def is_domain_partially_in_domain(v1, v0): 
    # checks if domain v1 is partially inside domain v0
    # domain v1 corners (TL = top left, TR = top right, BL = bottom left, BR = botom right)
    is_partially_in = False
    
    v0_L, v0_R = v0[0], v0[1]
    v1_L, v1_R = v1[0], v1[1]
    
    if is_in_domain(v1_L, v1_R, v0_L, v0_R):
        is_partially_in = True
    return is_partially_in


class privtree():
    # simple tree parameters
    # eps : dp param
    # theta : 50 #min count per domain
    # notice that the data inputed into this mechanism all need preprocess, which means K = data.shape[1]

    def __init__(self, rho, theta=0, domain_margin=1e-2, max_splits=None, seed=109):
        self.rho = rho
        self.theta = theta 
        self.domain_margin = domain_margin
        self.max_splits = max_splits
        self.prng = np.random.default_rng(seed)

    def fit(self, data):
        self.split = []
        self.has_nan = np.isnan(data).any()
        if self.has_nan:
            raise Warning('Data contains NaN')

        if data.ndim > 1:
            K = data.shape[1]
            self.lam, self.delta = self.calculate_param(self.rho, K)
            for i in range(data.shape[1]):
                x = data[:, i]
                self.split.append(self.tree_main(x, self.lam*K, self.delta*K, self.max_splits))
        else:
            K = 1
            self.lam, self.delta = self.calculate_param(self.rho, K)
            self.split = self.tree_main(data, self.lam, self.delta, self.max_splits)
    

    def transform(self, data):
        if data.ndim > 1:
            transformed_data = np.empty_like(data, dtype=float if self.has_nan else int)
            for i in range(data.shape[1]):
                x = data[:, i]
                transformed_data[:, i] = self.transform_data(x, self.split[i])
        else:
            transformed_data = self.transform_data(data, self.split)
        
        return transformed_data


    def fit_transform(self, data):
        self.split = []
        self.has_nan = np.isnan(data).any()
        if self.has_nan:
            warnings.warn("Data contains NaN", UserWarning)

        if data.ndim > 1:
            transformed_data = np.empty_like(data, dtype=float if self.has_nan else int)
            K = data.shape[1]
            self.lam, self.delta = self.calculate_param(self.rho, K)
            for i in range(data.shape[1]):
                x = data[:, i]
                self.split.append(self.tree_main(x, self.lam*K, self.delta*K, self.max_splits))
                transformed_data[:, i] = self.transform_data(x, self.split[i])
        else:
            K = 1
            self.lam, self.delta = self.calculate_param(self.rho, K)
            self.split = self.tree_main(data, self.lam, self.delta, self.max_splits)
            transformed_data = self.transform_data(data, self.split)

        return transformed_data


    def inverse_transform(self, data):
        inversed_data = np.empty_like(data, dtype=float)

        if data.ndim > 1:
            for i in range(data.shape[1]):
                inversed_data[:, i] = self.inverse_data(data[:,i], self.split[i])
        else:
            inversed_data = self.inverse_data(data, self.split)
            
        return inversed_data


    def tree_main(self, x_input, lam, delta, max_splits=None):
        domains = []
        unvisited_domains = []
        counts = []
        noisy_counts = []
        tree_depth = 0
        if max_splits is None:
            max_splits = np.inf

        x = x_input[~np.isnan(x_input)]

        x_min, x_max = np.min(x), np.max(x)
        v0 = [x_min, x_max + self.domain_margin] 
        unvisited_domains.append(v0)

        # create subdomains where necessary
        while unvisited_domains and len(domains) < max_splits: # while unvisited_domains is not empty
            for unvisited in unvisited_domains:
                # calculate count and noisy count
                count = count_in_domain(x, unvisited)
                b = count - (delta*tree_depth)
                b = max(b, (self.theta - delta))
                noisy_b = b + self.prng.laplace(loc=0, scale=lam)

                if (noisy_b > self.theta): #split if condition is met
                    v1, v2 = get_domain_subdomains(unvisited)
                    unvisited_domains.append(v1)
                    unvisited_domains.append(v2)
                    unvisited_domains.remove(unvisited)
                    tree_depth += 1
                else:
                    unvisited_domains.remove(unvisited)
                    counts.append(count)
                    noisy_counts.append(noisy_b)
                    domains.append(unvisited)

        return sorted(domains, key=lambda t: t[0])


    def transform_data(self, data_col, domains):
        is_nan = np.isnan(data_col)

        conditions = [(data_col >= a) & (data_col < b) for a, b in domains]
        conditions[0] = (data_col < domains[0][1])
        conditions[-1] = (data_col >= domains[-1][0])

        choices = list(range(len(domains)))
        result = np.select(conditions, choices, default=-1)

        if self.has_nan:
            result = result.astype(float)
            result[is_nan] = data_col[is_nan]

        return result
    
    def inverse_data(self, data_col, domains):
        return np.array([self.prng.uniform(domains[int(i)][0], domains[int(i)][1]) for i in data_col])


    def calculate_param(self, rho, K):
        beta = 2 
        lam = (2*beta - 1)/(beta - 1) * np.sqrt(K/(2*rho))
        delta = lam * np.log(beta)

        return lam, delta
        