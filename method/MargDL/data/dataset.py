import torch 
import numpy as np 
import pandas as pd 
import torch.nn.functional as F
import itertools


class Dataset():
    def __init__(self, df, domain, device):
        self.domain = domain
        self.df = df
        self.device = device
        self.num_records = df.shape[0]
        self.est_num_records = 0  # estimated number of records

        self.num_classes = list(domain.values())
        self.column_name = list(domain.keys())

    
    def update_est_records(self, num, rho):
        if rho is not None:
            if self.est_num_records == 0:
                self.est_num_records = num
                self.acc_weight = np.sqrt(rho)
            else:
                self.est_num_records = (self.acc_weight * self.est_num_records + np.sqrt(rho) * num)/(self.acc_weight + np.sqrt(rho))
                self.acc_weight += np.sqrt(rho)
        else:
            pass 

    def marginal_query(self, attr_tuple, rho = None, shape = 'simple', scale=True, update_records = False):
        assert (
            shape in ['matrix', 'simple']
        ), "Please provide proper shape request"

        data = self.df[list(attr_tuple)]
        data = data.dropna()
        data = data.to_numpy()
        
        bins = [np.arange(self.domain[attr] + 1) for attr in attr_tuple]

        joint_prob = np.histogramdd(data, bins=bins)[0]

        if rho is not None:
            joint_prob += np.random.normal(loc=0, scale=1/np.sqrt(2*rho), size=joint_prob.shape)

        if update_records:
            self.update_est_records(np.sum(joint_prob), rho)

        joint_prob = np.clip(joint_prob, 0, np.inf)
        if scale:
            joint_prob = joint_prob / np.sum(joint_prob)

        if shape == 'matrix':
            return joint_prob
        elif shape == 'simple':
            return joint_prob.flatten()


    def obtain_all_query(self, rho=None, scale=True, order=2, update_records=False):
        '''
        This function return all marginal query, used for ablation study
        The answer are in order !!!
        '''
        all_attr_tuple = list(itertools.combinations(self.domain.keys(), order))
        res = []

        if rho is not None:
            rho_each = rho/len(all_attr_tuple)
        else:
            rho_each=rho

        for attr_tuple in all_attr_tuple:
            ans = self.marginal_query(attr_tuple, rho=rho_each, scale=scale, update_records=update_records)
            res += list(ans) 
        
        return res

    def obtain_all_query_index(self, order=2):
        '''
        This function return the index of all 2-way marginal query
        '''
        if order == 1:
            total_dim = sum(self.domain.values())
            return list(np.arange(total_dim))
        elif order == 2:
            attr_ranges = {}
            start = 0
            for attr, num_classes in self.domain.items():
                attr_ranges[attr] = range(start, start + num_classes)
                start += num_classes
            
            attr_list = list(self.domain.keys())
            return [
                (col1, col2)
                for attr1, attr2 in itertools.combinations(attr_list, 2)
                for col1, col2 in itertools.product(attr_ranges[attr1], attr_ranges[attr2])
            ]
        else:
            raise NotImplementedError

    def query_from_indices(self, indices, rho=None, scale=True):
        if not hasattr(self, '_idx_to_attr_class'):
            self._idx_to_attr_class = {}
            start = 0
            for attr, num_classes in self.domain.items():
                for i in range(num_classes):
                    self._idx_to_attr_class[start + i] = (attr, i)
                start += num_classes
        
        mask = np.ones(len(self.df), dtype=bool)
        for idx in indices:
            attr, class_value = self._idx_to_attr_class[idx]
            mask &= (self.df[attr].values == class_value)
        count = float(mask.sum())
        
        # add noise
        if rho is not None:
            count += np.random.normal(loc=0, scale=1/np.sqrt(2*rho))
            count = np.clip(count, 0, np.inf)
        
        # scale
        if scale:
            return count / len(self.df)
        return count

    def reverse_to_ordinal(self, one_hot_tensor):
        cum_num_classes = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(self.num_classes, dtype=torch.int64), dim=0)])

        ordinal_tensors = []
        for i in range(len(self.num_classes)):
            start, end = cum_num_classes[i], cum_num_classes[i + 1]
            probs = one_hot_tensor[:, start:end] 

            assert not torch.isnan(probs).any().item(), "probs contains NaN"
            assert not torch.isinf(probs).any().item(), "probs contains Inf"
            assert not (probs.sum(dim=-1) == 0).any().item(), 'invalid output (all zero)'
            assert not (probs < 0).any().item(), 'invalid output (negative prob)'

            idxs = torch.multinomial(probs, num_samples=1).squeeze(dim=-1)  
            ordinal_tensors.append(idxs)

        ordinal_tensor = torch.stack(ordinal_tensors, dim=1)
        
        return ordinal_tensor
    
    def reverse_data(self, data, num_samples, preprocesser=None, path=None):
        '''
        reverse data into ordinal type and save them in the specified path
        '''
        ordinal_data = self.reverse_to_ordinal(data).cpu().numpy()
        idxs = np.random.choice(ordinal_data.shape[0], size=num_samples, replace=False)
        ordinal_data = ordinal_data[idxs]
        
        if preprocesser is not None:
            preprocesser.reverse_data(ordinal_data, path)
        return ordinal_data
    
    # def update_num_records_est(self, rho, est_value):
    #     if not self.records_est:
    #         self._num_records = est_value 
    #         self.est_param += np.sqrt(rho)
    #         self.records_est = True
    #     else:
    #         self._num_records = (self.est_param * self._num_records + np.sqrt(rho) * est_value)/(self.est_param + np.sqrt(rho))
    #         self.est_param += np.sqrt(rho)

    # @property
    # def num_records(self):
    #     if not self.records_est:
    #         print('You are using precise value of record number!!!!')
    #         print('We suggest you to initialize model by one-way marginals and estimate record number with DP')
    #     return self._num_records

        