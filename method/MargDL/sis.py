import numpy as np
import pandas as pd
from copy import deepcopy
from method.MargDL.data.dataset import Dataset
from method.privsyn.PrivSyn.privsyn import PrivSyn

class SIS:
    def __init__(self, args, df: pd.DataFrame, domain: dict, device=None):
        """
        df: pandas DataFrame (not directly used in SIS sampling)
        domain: dict mapping variable names to number of categories (0 to x-1)
        device: placeholder for compatibility
        """
        self.args = args
        self.dataset = Dataset(df, domain, device)
        self.domain = domain
        self.var_names = list(domain.keys())

    def obtain_marginal(self, rho):
        """
        Placeholder: assumes external computation of marginals.
        Expected output: list of tuples
            [(('A', 'B'), flat_counts_list, weight), ...]
        """
        marginals = PrivSyn.two_way_marginal_selection(self.dataset.df, self.dataset.domain, 0.1*rho, 0.8*rho)
        self.selected_marginals = [
            (   
                marginals[i], 
                self.dataset.marginal_query(marginals[i], None), 
                None
            )
            for i in range(len(marginals))
        ]
        return 0


    def generate(self, num_samples):
        """
        Generate an initial dataset of shape (N, p) satisfying given 2-way marginals.
        selected_marginals: list of ( (var1, var2), flat_probs, _ )
                           flat_probs is normalized (sums to 1)
        """
        N = num_samples

        # Reconstruct 2D marginal count arrays from normalized probs
        marg_counts = {}
        for (v1, v2), flat_probs, _ in self.selected_marginals:
            d1, d2 = self.domain[v1], self.domain[v2]
            pmat = np.array(flat_probs).reshape(d1, d2)
            # convert to integer counts
            counts = np.round(pmat * N).astype(int)
            # adjust sum to exactly N
            diff = counts.sum() - N
            if diff != 0:
                # compute fractional parts and sort
                frac = (pmat * N) - counts
                # flatten
                idxs = np.dstack(np.unravel_index(np.argsort(frac.ravel()), (d1, d2)))[0]
                if diff > 0:
                    # too many, reduce cells with smallest frac
                    for i in range(diff):
                        x, y = idxs[i]
                        counts[x, y] -= 1
                else:
                    # too few, add to largest frac
                    for i in range(-diff):
                        x, y = idxs[::-1][i]
                        counts[x, y] += 1
            # store both directions
            marg_counts[(v1, v2)] = counts.copy()
            marg_counts[(v2, v1)] = counts.T.copy()

        # Prepare remnant marginals for SIS
        remaining = deepcopy(marg_counts)
        sampled = np.zeros((N, len(self.var_names)), dtype=int)

        # SIS sampling
        for i in range(N):
            row = {}
            for j, var in enumerate(self.var_names):
                size = self.domain[var]
                weights = np.ones(size, dtype=float)
                # accumulate weights from existing marginals
                for prev_var, prev_val in row.items():
                    if (var, prev_var) in remaining:
                        w = remaining[(var, prev_var)][:, prev_val]
                        # clip negatives to zero
                        w = np.clip(w, 0, None)
                        weights *= w
                # if all weights zero (or negative), fallback to uniform
                if weights.sum() <= 0:
                    weights = np.ones(size, dtype=float)
                probs = weights / weights.sum()
                val = np.random.choice(size, p=probs)

                row[var] = val
                sampled[i, j] = val

                # update remaining marginals
                for prev_var, prev_val in row.items():
                    if prev_var == var:
                        continue
                    if (var, prev_var) in remaining:
                        remaining[(var, prev_var)][val, prev_val] -= 1
                        remaining[(prev_var, var)][prev_val, val] -= 1

        return sampled
    

    def sample(self, num_samples, preprocesser=None, parent_dir=None):
        syn_data = self.generate(num_samples=num_samples)

        preprocesser.reverse_data(syn_data, parent_dir)
        return syn_data 
    



def sis_main(args, df, domain, rho, **kwargs):
    generator = SIS(
        args,
        df,
        domain,
        device='cpu'
    )
    generator.obtain_marginal(rho)

    return {'MargDL_generator': generator}