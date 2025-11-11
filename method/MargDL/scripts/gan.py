import torch
import torch.nn.functional as F
import numpy as np
import itertools
import copy
import os
import math
import time
import pandas as pd
from inspect import isfunction
from collections import deque
from method.MargDL.scripts.denoiser import *
from method.MargDL.scripts.graph import *


class MargGAN(nn.Module):
    def __init__(self, config, domain, device='cuda:0', **kwargs):
        super(MargGAN, self).__init__()

        self.config = config
        self.device = device
        self.parent_dir = kwargs.get('parent_dir', None)

        self.column_dims = domain  # Dictionary storing column name -> one-hot dimension
        self.num_classes = np.array(list(domain.values()))
        self.column_name = np.array(list(domain.keys()))
        self.marginals = []  # List to store (tuple a, array b, weight)

        self.cum_num_classes = np.concatenate((np.array([0]), np.cumsum(self.num_classes)))

        self.batch_size = self.config['train']['batch_size']
        self.z = None
        self.queries = None 
        self.real_answers = None
        self.resample = kwargs.get('resample', False)
        self.sample_type = kwargs.get('sample_type', 'direct')

        self.model = Generator(
            embedding_dim=self.config['model_params']['data_dim'],
            gen_dims=self.config['model_params']['d_layers'],
            data_dim=self.config['model_params']['data_dim']
        ).to(self.device)
    
    def reset_model(self):
        del self.model 
        self.model = Generator(
            embedding_dim=self.config['model_params']['data_dim'],
            gen_dims=self.config['model_params']['d_layers'],
            data_dim=self.config['model_params']['data_dim']
        ).to(self.device)


    def _find_marginal_index(self, marginals):
        index = []
        answer = []
        size = []
        weight = []

        for i, (marg, matrix, w) in enumerate(marginals):
            start_idx = [self.cum_num_classes[np.where(self.column_name == col)[0][0]] for col in marg]
            end_idx = [start + self.column_dims[col] for start, col in zip(start_idx, marg)]
            iter_list = [range(a,b) for (a, b) in zip(start_idx, end_idx)]

            index += list(itertools.product(*iter_list))
            answer += matrix.tolist()
            size += [1/matrix.size for _ in range(matrix.size)]
            weight += [w for _ in range(matrix.size)]

        answer = torch.tensor(answer, dtype=torch.float64, device=self.device)
        weight = torch.tensor(weight, dtype=torch.float64, device=self.device)
        return index, answer, size, weight


    def _merge_marginals(self, marginals):
        merged = {}
        for i, (name, matrix, weight) in enumerate(marginals):
            if name not in merged:
                merged[name] = (matrix * weight, weight)
            else:
                weighted_sum, total_weight  = merged[name]
                merged[name] = (weighted_sum + matrix * weight, total_weight + weight)
        
        result = []
        for name, (weighted_sum, total_weight) in merged.items():
            avg_matrix = weighted_sum / total_weight
            result.append((name, avg_matrix, total_weight))
    
        return result
    

    def store_marginals(self, marginals, **kwargs):
        '''
        tansfer marginal to queries and store them
        '''
        merged_marginals = self._merge_marginals(marginals)
        marginal_list = [marg_list[0] for marg_list in merged_marginals]

        self.jt = generate_junction_tree(list(self.column_dims.keys()), marginal_list)
        self.queries, self.real_answers, self.query_size, self.query_weight = self._find_marginal_index(merged_marginals) 

        K = len(self.queries)
        D = int(self.cum_num_classes[-1])
        self.Q_mask = torch.zeros((K, D), device=self.device, dtype=torch.float32)
        for k, q in enumerate(self.queries):
            self.Q_mask[k, q] = 1.0


    @torch.no_grad()
    def _uniform_sample(self):
        '''
        initialize a uniform distributed tensor x_T, as the start of the posterior process
        '''
        if (self.z is None) or self.resample:
            z_oh = []
            for i in range(len(self.cum_num_classes)-1):
                start = self.cum_num_classes[i]
                end = self.cum_num_classes[i+1]

                probs = torch.tensor([1/(end-start) for _ in range(end-start)]).to(self.device)
                idxs = torch.multinomial(probs, self.batch_size, replacement=True)
                idxs = F.one_hot(idxs, num_classes=(end - start))
                idxs = torch.clamp(idxs, 1e-30, 1-(end-start)*1e-30)
                z_oh.append(idxs.float())

            self.z = torch.cat(z_oh, dim=1)

            # self.z = torch.randn((self.batch_size, self.config['model_params']['data_dim']), device=self.device)

        return self.z
    
    def _predict_x(self, xt):
        output = self.model(xt)
        data = []
        for i in range(len(self.cum_num_classes)-1):
            st = self.cum_num_classes[i]
            ed = self.cum_num_classes[i+1]
            logits = output[:, st:ed].log_softmax(-1)
            data.append(torch.clamp(logits, min=-30, max=0.0))
            # data.append(logits)

        return torch.cat(data, dim=1) 


    def _compute_loss(self, Q, ans_weight, real_ans):
        x_pred = self._predict_x(self._uniform_sample())  
        
        S = x_pred @ Q.T # x_pred is a logits now
        syn_ans = S.exp().mean(dim=0)  

        loss  = (ans_weight * (syn_ans - real_ans)**2).sum()
        return loss
  

    def train_model(self, lr, iterations, save_loss=False, path_prefix = None, **kwargs):
        self.model.train()
        self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=lr
            )
        if save_loss:
            loss_tracker = pd.DataFrame(columns=['iter', 'loss'])
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, iterations, eta_min=1e-6)

        for iter in range(iterations):

            self.optimizer.zero_grad()

            loss = self._compute_loss(self.Q_mask, self.query_weight, self.real_answers)
            loss.backward()
            self.optimizer.step()

            if save_loss:
                loss_tracker.loc[len(loss_tracker)] = [iter, loss.item()]
        
        torch.save(self.model.state_dict(), os.path.join(self.parent_dir, 'model.pt'))
        if save_loss:
            loss_tracker.to_csv(os.path.join(self.parent_dir, f'{path_prefix}_loss.csv'))
        


    def _sample_logits(self):
        input = self._uniform_sample()
        x = self._predict_x(input)
        return x

    def sample(self, num_samples):
        self.model.eval()
        if self.sample_type == 'graphical':
            logits = self._sample_logits().exp()
            res = graph_sample(logits, self.jt, self.column_name, self.cum_num_classes, self.device, num_samples=num_samples)
            return res 
        else:
            logits = self._sample_logits().exp()
            idx = torch.randint(0, logits.shape[0], size=(num_samples,))
            logits = logits[idx, :]
            res = []
            for i in range(len(self.num_classes)):
                start, end = self.cum_num_classes[i], self.cum_num_classes[i + 1]
                probs = logits[:, start:end] 

                idxs = torch.multinomial(probs, num_samples=1).squeeze(dim=-1)  
                res.append(idxs)

            res = torch.stack(res, dim=1)
            return res.detach().cpu().numpy()


    def _map_to_marginal(self, x0_pred, marginal):
        start_idx = [self.cum_num_classes[np.where(self.column_name == col)[0][0]] for col in marginal]
        end_idx = [start + self.column_dims[col] for start, col in zip(start_idx, marginal)]

        z_splits = [x0_pred[:, start:end] for start, end in zip(start_idx, end_idx)]

        input_dims = ','.join(f'b{chr(105 + i)}' for i in range(len(z_splits)))
        output_dims = ''.join(f'{chr(105 + i)}' for i in range(len(z_splits)))
        einsum_str = f'{input_dims}->b{output_dims}'
        joint_prob = torch.einsum(einsum_str, *z_splits).mean(dim=0).to(self.device)

        return joint_prob.detach().cpu().numpy().flatten()


    @torch.no_grad()
    def obtain_sample_marginals(self, marginals, **kwargs):
        self.model.eval()
        logits = self._sample_logits()
        res = []
        if self.sample_type == 'graphical':
            samples = graph_sample(logits.exp(), self.jt, self.column_name, self.cum_num_classes, num_samples = 102400)
            for marginal in marginals:
                idx = [np.where(self.column_name == marg)[0] for marg in marginal]
                data = samples[:, idx]
                bins = [np.arange(self.column_dims[attr] + 1) for attr in marginal]
                joint_prob = np.histogramdd(data, bins=bins)[0].flatten()
                res.append(joint_prob)
        else:
            for marginal in marginals:
                res.append(self._map_to_marginal(logits.exp(), marginal))
        return res


    # Used for ablation

    def _merge_queries(self, queries):
        q = []
        ans = []
        w = []
        for q_idx, answer, weight in queries:
            q.append(list(q_idx))
            ans.append(answer)
            w.append(weight)
        
        ans = torch.tensor(ans, dtype=torch.float64, device=self.device)
        w = torch.tensor(w, dtype=torch.float64, device=self.device)

        return q, ans, w

    def store_queries(self, queries, **kwargs):
        '''
        store queries for training
        '''
        self.queries, self.real_answers, self.query_weight = self._merge_queries(queries)

        K = len(self.queries)
        D = int(self.cum_num_classes[-1])
        print(f'Q len: {D}')
        self.Q_mask = torch.zeros((K, D), device=self.device, dtype=torch.float32)
        for k, q in enumerate(self.queries):
            self.Q_mask[k, q] = 1.0

    @torch.no_grad()
    def obtain_sample_queries(self, queries, **kwargs):
        self.model.eval()
        x0_pred = self._sample_logits().exp()
        res = []

        for q in queries:
            res.append(
                x0_pred[:, list(q)].prod(dim=1).mean().item()
            )

        return res