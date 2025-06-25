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

def prob_to_softmax_onehot(x, num_classes, tau=1.0):
    splits = torch.split(x, num_classes.tolist(), dim=1)
    one_hot_splits = [
        F.softmax(split, dim=1) for split in splits
    ]

    one_hot_x = torch.cat(one_hot_splits, dim=1)
    return one_hot_x 


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


    def find_query_index(self, marginals):
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


    def merge_marginals(self, marginals, enhance_weight):
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
    

    def store_marginals(self, marginals, enhance_weight=1.0, **kwargs):
        '''
        tansfer marginal to queries and store them
        '''
        merged_marginals = self.merge_marginals(marginals, enhance_weight)
        marginal_list = [marg_list[0] for marg_list in merged_marginals]

        self.jt = generate_junction_tree(list(self.column_dims.keys()), marginal_list)
        self.queries, self.real_answers, self.query_size, self.query_weight = self.find_query_index(merged_marginals) 

        K = len(self.queries)
        D = int(self.cum_num_classes[-1])
        self.Q_mask = torch.zeros((K, D), device=self.device, dtype=torch.float32)
        for k, q in enumerate(self.queries):
            self.Q_mask[k, q] = 1.0

    
    def initialize_logits(self, marginals=None):
        logits_list = []
        if marginals:
            marginals_dict = {marg[0] : marg[1] for marg in marginals}
        else:
            marginals_dict = {}

        for col_name, col_size in zip(self.column_name, self.num_classes):
            if col_name in marginals_dict:
                arr = marginals_dict[col_name]
                if len(arr) != col_size:
                    raise ValueError('Invalid one-way marginal')
            else:
                arr = np.ones(col_size, dtype=np.float32) / col_size

            logits_list.append(torch.tensor(arr, device=self.device))

        self.init_logits = torch.cat(logits_list, dim=0)

        return self.init_logits

    @torch.no_grad()
    def uniform_sample(self):
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
    
    def predict_x(self, xt):
        output = self.model(xt)
        data = []
        for i in range(len(self.cum_num_classes)-1):
            st = self.cum_num_classes[i]
            ed = self.cum_num_classes[i+1]
            logits = output[:, st:ed].log_softmax(-1)
            data.append(torch.clamp(logits, min=-30, max=0.0))
            # data.append(logits)

        return torch.cat(data, dim=1) 


    def compute_loss(self, Q, ans_weight, real_ans):
        x_pred = self.predict_x(self.uniform_sample())  
        
        S = x_pred @ Q.T
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

            loss = self.compute_loss(self.Q_mask, self.query_weight, self.real_answers)
            loss.backward()
            self.optimizer.step()

            if save_loss:
                loss_tracker.loc[len(loss_tracker)] = [iter, loss.item()]
        
        torch.save(self.model.state_dict(), os.path.join(self.parent_dir, 'model.pt'))
        if save_loss:
            loss_tracker.to_csv(os.path.join(self.parent_dir, f'{path_prefix}_loss.csv'))
        


    def sample_logits(self):
        input = self.uniform_sample()
        x = self.predict_x(input)
        return x

    def sample(self, num_samples):
        if self.sample_type == 'graphical':
            logits = self.sample_logits().exp()
            res = graph_sample(logits, self.jt, self.column_name, self.cum_num_classes, self.device, num_samples=num_samples)
            return res 
        else:
            logits = self.sample_logits().exp()
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


    def map_to_marginal(self, x0_pred, marginal):
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
        logits = self.sample_logits()
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
                res.append(self.map_to_marginal(logits.exp(), marginal))
        return res
        








    # def train_model_index(self, info, lr, iterations, sub_iterations, max_threshold=500, **kwargs):
    #     track = pd.DataFrame(columns=[f'loss'])
    #     select_time = 0.0
    #     train_time = 0.0

    #     self.model.train()
    #     self.optimizer = torch.optim.Adam(
    #             self.model.parameters(), 
    #             lr=lr
    #         )
    #     # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, iterations, eta_min=1e-6)

    #     for iter in range(iterations):

    #         start_time = time.time()
    #         if math.isinf(max_threshold) or True:
    #             topk_indices = None 

    #             ans_weight = self.query_weight
    #             real_ans = self.real_answers
    #             Q = self.Q_mask

    #             # K = len(self.queries)
    #             # D = int(self.cum_num_classes[-1])
    #             # Q = torch.zeros((K, D), device=self.device, dtype=torch.float32)
    #             # for k, q in enumerate(self.queries):
    #             #     Q[k, q] = 1.0
                
    #         else:
    #             errors = self.compute_error_index()
    #             k = min(len(errors), max_threshold)
    #             _, topk_indices = torch.topk(errors, k, largest=True, sorted=False)

    #             ans_weight = torch.stack([self.query_weight[i.item()] for i in topk_indices], dim=0).to(self.device)
    #             real_ans = torch.stack([self.real_answers[i.item()] for i in topk_indices], dim=0).to(self.device)
    #             Q = self.Q_mask[topk_indices]

    #             # K = len(topk_indices)
    #             # D = int(self.cum_num_classes[-1])
    #             # Q = torch.zeros((K, D), device=self.device, dtype=torch.float32)
    #             # sel_queries = [self.queries[i.item()] for i in topk_indices]
    #             # for k, q in enumerate(sel_queries):
    #             #     Q[k, q] = 1.0

    #         # if iter == 0:
    #         #     errors = self.compute_error_index()
    #         #     print('start loss:', errors.sum().item())

    #         end_time = time.time()
    #         select_time += (end_time - start_time)

    #         for _ in range(sub_iterations):
    #             self.optimizer.zero_grad()

    #             start_time = time.time()
    #             loss = self.compute_loss_index(Q, ans_weight, real_ans)
    #             loss.backward()
    #             self.optimizer.step()
    #             end_time = time.time()
    #             train_time += (end_time - start_time)
                
    #         # print(f'iter: {iter}/{iterations}', end='\r')
    #         track.loc[len(track)] = [loss.item()]
    #         # if iter == iterations - 1:
    #         #     errors = self.compute_error_index()
    #         #     print('end loss:', errors.sum().item())

    #     track.to_csv(os.path.join(self.parent_dir, f'{info}_loss_track.csv'))
    #     torch.save(self.model.state_dict(), os.path.join(self.parent_dir, 'model.pt'))
    #     print('select time:', select_time)
    #     print('train time:', train_time)



    # def predict_x(self, xt):
    #     output = self.model(xt)
    #     data = []
    #     for i in range(len(self.cum_num_classes)-1):
    #         st = self.cum_num_classes[i]
    #         ed = self.cum_num_classes[i+1]
    #         logits = output[:, st:ed].log_softmax(-1)
    #         data.append(torch.clamp(logits, min=-30, max=0.0))
    #         # data.append(logits)

    #     return torch.cat(data, dim=1) 
    

    # def compute_loss(self):
    #     # input = self.uniform_sample()
    #     # x_pred = self.predict_x(input)

    #     # syn_attrs = [x_pred[:, q_id] for q_id in self.queries]
    #     # syn_answers = [syn_attr.prod(-1).mean(axis=0) for syn_attr in syn_attrs]
    #     # true_answers = [self.real_answers[i] for i in range(len(self.queries))]

    #     # return sum(self.query_weight[i] * (syn_answers[i] - true_answers[i])**2 for i in range(len(true_answers)))

    #     x_pred = self.predict_x(self.uniform_sample())       # (batch, D)
    #     log_x = x_pred.log()                                  # (batch, D)

    #     S     = log_x @ self.Q_mask.T                         # (batch, K)
    #     syn   = S.exp().mean(dim=0)                           # (K,)
    #     w = self.query_weight.to(self.device)            # (K,)
    #     t = self.real_answers.to(self.device)  # (K,)

    #     loss  = (w * (syn - t)**2).sum()
    #     return loss


    # def compute_loss_index(self, id):
    #     # input = self.uniform_sample()
    #     # x_pred = self.predict_x(input)

    #     # if id is not None:
    #     #     selected_queries = [self.queries[i.item()] for i in id]
    #     #     selected_weight = [self.query_weight[i.item()] for i in id]
    #     #     true_answers = [self.real_answers[i] for i in id]
    #     # else:
    #     #     selected_queries = self.queries
    #     #     selected_weight = self.query_weight
    #     #     true_answers = [self.real_answers[i] for i in range(len(self.real_answers))]

    #     # syn_attrs = [x_pred[:, q_id] for q_id in selected_queries]
    #     # syn_answers = [syn_attr.prod(-1).mean(axis=0) for syn_attr in syn_attrs]

    #     # return sum(selected_weight[i] * (syn_answers[i] - true_answers[i])**2 for i in range(len(true_answers)))
    
    #     x_pred = self.predict_x(self.uniform_sample())       # (batch, D)
        
    #     if id is not None:
    #         idx = id.view(-1)
    #         Q   = self.Q_mask[idx]                           # (K_sel, D)
    #         w   = torch.stack([self.query_weight[i.item()] for i in idx], dim=0).to(self.device)
    #         t   = torch.stack([self.real_answers[i.item()] for i in idx], dim=0).to(self.device)
    #     else:
    #         Q   = self.Q_mask                                # (K, D)
    #         w = self.query_weight.to(self.device)            # (K,)
    #         t = self.real_answers.to(self.device)            # (K,)
        
    #     if torch.isinf(x_pred).any(): print('inf detected in output')
    #     if torch.isnan(x_pred).any(): print('nan detected in output')

    #     log_x = x_pred.log()                                  # (batch, D)

    #     # S     = log_x @ Q.T                                   # (batch, K_sel)
    #     S = x_pred @ Q.T
    #     syn   = S.exp().mean(dim=0)                           # (K_sel,)

    #     if torch.isinf(syn).any(): print('inf detected in log-exp output')
    #     if torch.isnan(syn).any(): print('nan detected in log-exp output')

    #     loss  = (w * (syn - t)**2).sum()

    #     self.detect_nan(loss, 'loss')

    #     return loss

    
    # @torch.no_grad()
    # def compute_error_index(self):
    #     # input = self.uniform_sample()
    #     # x_pred = self.predict_x(input)

    #     # syn_attrs = [x_pred[:, q_id] for q_id in self.queries]
    #     # syn_answers = [syn_attr.prod(-1).mean(axis=0) for syn_attr in syn_attrs]
    #     # true_answers = [self.real_answers[i] for i in range(len(self.queries))]

    #     # return torch.tensor([self.query_weight[i] * (syn_answers[i] - true_answers[i])**2 for i in range(len(true_answers))], dtype=torch.float32, device=self.device)

    #     x_pred = self.predict_x(self.uniform_sample())       # (batch, D)
    #     # log_x = x_pred.log()                                  # (batch, D)
    #     S     = x_pred @ self.Q_mask.T                         # (batch, K)
    #     syn   = S.exp().mean(dim=0)                           # (K,)
    #     w     = self.query_weight.to(self.device, dtype=syn.dtype)
    #     t     = self.real_answers.to(self.device, dtype=syn.dtype)
    #     errors = w * (syn - t)**2                             # (K,)
    #     return errors






# class MargGANBoost(nn.Module):
#     def __init__(self, config, domain, device='cuda:0', **kwargs):
#         super(MargGANBoost, self).__init__()

#         self.config = config
#         self.device = device
#         self.parent_dir = kwargs.get('parent_dir', None)

#         self.column_dims = domain  # Dictionary storing column name -> one-hot dimension
#         self.num_classes = np.array(list(domain.values()))
#         self.column_name = np.array(list(domain.keys()))
#         self.marginals = []  # List to store (tuple a, array b, weight)

#         self.cum_num_classes = np.concatenate((np.array([0]), np.cumsum(self.num_classes)))

#         self.batch_size = self.config['train']['batch_size']
#         self.boost_num = 3

#         self.z = [None]*self.boost_num
#         self.queries = None 
#         self.real_answers = None
#         self.resample = kwargs.get('resample', False)
#         self.current_output = None

#         for i in range(self.boost_num):
#             model_path = os.path.join(self.parent_dir, f'model_{i}.pt')
#             if os.path.exists(model_path):
#                 os.remove(model_path)


#     def reset_model(self, i=0):    
#         self.model = Generator(
#             embedding_dim=self.config['model_params']['data_dim'],
#             gen_dims=self.config['model_params']['d_layers'],
#             data_dim=self.config['model_params']['data_dim']
#         ).to(self.device)
        
#         if os.path.exists(os.path.join(self.parent_dir, f'model_{i}.pt')):
#             self.model.load_state_dict(torch.load(os.path.join(self.parent_dir, f'model_{i}.pt')))
    
#     def find_query_index(self, marginals):
#         index = []
#         answer = []
#         size = []
#         weight = []
#         for (marg, matrix, w) in marginals:
#             start_idx = [self.cum_num_classes[np.where(self.column_name == col)[0][0]] for col in marg]
#             end_idx = [start + self.column_dims[col] for start, col in zip(start_idx, marg)]
#             iter_list = [range(a,b) for (a, b) in zip(start_idx, end_idx)]
#             index += list(itertools.product(*iter_list))
#             answer += matrix.tolist()
#             size += [1/matrix.size for _ in range(matrix.size)]
#             weight += [w for _ in range(matrix.size)]
#         # return torch.tensor(index, device=self.device), torch.tensor(answer, device=self.device)
#         return index, torch.tensor(answer, dtype=torch.float64, device=self.device), size, torch.tensor(weight, dtype=torch.float64, device=self.device)


#     def store_marginals(self, marginals, enhance_weight=None, **kwargs):
#         '''
#         tansfer marginal to queries and store them
#         '''
#         self.marginal_list = [marg_list[0] for marg_list in marginals]
#         self.queries, self.real_answers, self.query_size, self.query_weight = self.find_query_index(marginals)
#         self.target_answers = self.real_answers.clone()

#     @torch.no_grad()
#     def update_margianls(self, res):
#         current_size = res.shape[0] 
#         for i in range(len(self.queries)):
#             current_ans = res[:, self.queries[i]].prod(-1).mean(axis=0)
#             self.target_answers[i] = ((current_size + self.batch_size) * self.real_answers[i] - current_size * current_ans)/self.batch_size
#         self.target_answers = torch.clamp(self.target_answers, 0, 1)

    
#     def initialize_logits(self, marginals=None):
#         logits_list = []
#         if marginals:
#             marginals_dict = {marg[0] : marg[1] for marg in marginals}
#         else:
#             marginals_dict = {}

#         for col_name, col_size in zip(self.column_name, self.num_classes):
#             if col_name in marginals_dict:
#                 arr = marginals_dict[col_name]
#                 if len(arr) != col_size:
#                     raise ValueError('Invalid one-way marginal')
#             else:
#                 arr = np.ones(col_size, dtype=np.float32) / col_size

#             logits_list.append(torch.tensor(arr, device=self.device))

#         self.init_logits = torch.cat(logits_list, dim=0)

#         return self.init_logits

#     @torch.no_grad()
#     def uniform_sample(self, i=0):
#         '''
#         initialize a uniform distributed tensor x_T, as the start of the posterior process
#         '''
#         if (self.z[i] is None) or self.resample:
#             z_oh = []
#             for j in range(len(self.cum_num_classes)-1):
#                 start = self.cum_num_classes[j]
#                 end = self.cum_num_classes[j+1]

#                 probs = torch.tensor([1/(end-start) for _ in range(end-start)]).to(self.device)
#                 idxs = torch.multinomial(probs, self.batch_size, replacement=True)
#                 idxs = F.one_hot(idxs, num_classes=(end - start))
#                 idxs = torch.clamp(idxs, 1e-30, 1-(end-start)*1e-30)
#                 z_oh.append(idxs.float())

#             self.z[i] = torch.cat(z_oh, dim=1)

#         return self.z[i]
    
#     def predict_x(self, xt):
#         logits = self.model(xt)
#         data = []
#         for i in range(len(self.cum_num_classes)-1):
#             st = self.cum_num_classes[i]
#             ed = self.cum_num_classes[i+1]
#             temp_data = logits[:, st:ed].softmax(-1)

#             assert not torch.isnan(temp_data).any().item(), "invalid output (NaN)"
#             assert not torch.isinf(temp_data).any().item(), "invalid output (Inf)"
#             assert not (temp_data.sum(dim=-1) == 0).any().item(), 'invalid output (all zero)'
#             assert not (temp_data < 0).any().item(), 'invalid output (negative prob)'

#             data.append(temp_data)
#         return torch.cat(data, dim=1) 
#         # return logits.exp()
    

#     def compute_loss(self, i=0):
#         input = self.uniform_sample(i=i)
#         x_pred = self.predict_x(input)

#         syn_attrs = [x_pred[:, q_id] for q_id in self.queries]
#         syn_answers = [syn_attr.prod(-1).mean(axis=0) for syn_attr in syn_attrs]
#         true_answers = [self.target_answers[i] for i in range(len(self.queries))]

#         return sum(self.query_weight[i] * (syn_answers[i] - true_answers[i])**2 for i in range(len(true_answers)))

#     def train_model(self, info, lr, iterations, output_loss = True, **kwargs):
#         self.current_output = None
#         for i in range(self.boost_num):
#             track = pd.DataFrame(columns=[f'loss'])
#             self.reset_model(i=i)
#             self.model.train()

#             self.optimizer = torch.optim.Adam(
#                     self.model.parameters(), 
#                     lr=lr
#                 )
#             # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, iterations, eta_min=1e-6)
#             for iter in range(iterations):
#                 self.optimizer.zero_grad()
#                 # self.anneal_enhace_weight(iter, iterations)

#                 loss = self.compute_loss(i)
#                 loss.backward()
#                 self.optimizer.step()
                
#                 # self.scheduler.step()
                
#                 if iter == 0:
#                     print('start loss:', loss.item())
#                 elif iter == iterations-1:
#                     print('end loss:', loss.item())

#                 track.loc[len(track)] = [loss.item()]

#             track.to_csv(os.path.join(self.parent_dir, f'{info}_loss_track_{i}.csv'))
#             torch.save(self.model.state_dict(), os.path.join(self.parent_dir, f'model_{i}.pt'))

#             self.model.eval()
#             new_res = self.predict_x(self.uniform_sample(i=i))
#             if self.current_output is None:
#                 self.current_output = new_res
#             else:
#                 self.current_output = torch.concat((self.current_output, new_res), dim=0)
            

#             self.update_margianls(self.current_output)

    
#     def sample(self, num_samples):
#         ind = torch.randint(0, self.current_output.shape[0], (num_samples,), dtype=torch.long)
#         res = self.current_output[ind]

#         return res


#     def map_to_marginal(self, x0_pred, marginal):
#         start_idx = [self.cum_num_classes[np.where(self.column_name == col)[0][0]] for col in marginal]
#         end_idx = [start + self.column_dims[col] for start, col in zip(start_idx, marginal)]

#         z_splits = [x0_pred[:, start:end] for start, end in zip(start_idx, end_idx)]

#         input_dims = ','.join(f'b{chr(105 + i)}' for i in range(len(z_splits)))
#         output_dims = ''.join(f'{chr(105 + i)}' for i in range(len(z_splits)))
#         einsum_str = f'{input_dims}->b{output_dims}'
#         joint_prob = torch.einsum(einsum_str, *z_splits).mean(dim=0).to(self.device)

#         return joint_prob.detach().cpu().numpy().flatten()

#     @torch.no_grad()
#     def obtain_sample_marginals(self, marginals, num_samples=1024):
#         x = self.sample(num_samples)
#         res = []
#         for marginal in marginals:
#             res.append(self.map_to_marginal(x, marginal))
        
#         return res
        



# class MargGANB(nn.Module):
#     def __init__(self, config, domain, device='cuda:0', **kwargs):
#         super(MargGANB, self).__init__()

#         self.config = config
#         self.device = device
#         self.parent_dir = kwargs.get('parent_dir', None)

#         self.column_dims = domain  # Dictionary storing column name -> one-hot dimension
#         self.num_classes = np.array(list(domain.values()))
#         self.column_name = np.array(list(domain.keys()))
#         self.marginals = []  # List to store (tuple a, array b, weight)

#         self.cum_num_classes = np.concatenate((np.array([0]), np.cumsum(self.num_classes)))

#         self.batch_size = self.config['train']['batch_size']
#         self.boost_num = 5

#         self.z = [None]*self.boost_num
#         self.queries = None 
#         self.real_answers = None
#         self.resample = kwargs.get('resample', False)
#         self.current_output = None

#         for i in range(self.boost_num):
#             model_path = os.path.join(self.parent_dir, f'model_{i}.pt')
#             if os.path.exists(model_path):
#                 os.remove(model_path)


#     def reset_model(self, i=0):    
#         self.model = Generator(
#             embedding_dim=self.config['model_params']['data_dim'],
#             gen_dims=self.config['model_params']['d_layers'],
#             data_dim=self.config['model_params']['data_dim']
#         ).to(self.device)
        
#         if os.path.exists(os.path.join(self.parent_dir, f'model_{i}.pt')):
#             self.model.load_state_dict(torch.load(os.path.join(self.parent_dir, f'model_{i}.pt')))
    
#     def find_query_index(self, marginals):
#         index = []
#         answer = []
#         size = []
#         weight = []
#         for (marg, matrix, w) in marginals:
#             start_idx = [self.cum_num_classes[np.where(self.column_name == col)[0][0]] for col in marg]
#             end_idx = [start + self.column_dims[col] for start, col in zip(start_idx, marg)]
#             iter_list = [range(a,b) for (a, b) in zip(start_idx, end_idx)]
#             index += list(itertools.product(*iter_list))
#             answer += matrix.tolist()
#             size += [1/matrix.size for _ in range(matrix.size)]
#             weight += [w for _ in range(matrix.size)]
#         # return torch.tensor(index, device=self.device), torch.tensor(answer, device=self.device)
#         return index, torch.tensor(answer, dtype=torch.float64, device=self.device), size, torch.tensor(weight, dtype=torch.float64, device=self.device)


#     def store_marginals(self, marginals, enhance_weight=None, **kwargs):
#         '''
#         tansfer marginal to queries and store them
#         '''
#         self.marginal_list = [marg_list[0] for marg_list in marginals]
#         self.queries, self.real_answers, self.query_size, self.query_weight = self.find_query_index(marginals)

    
#     def initialize_logits(self, marginals=None):
#         logits_list = []
#         if marginals:
#             marginals_dict = {marg[0] : marg[1] for marg in marginals}
#         else:
#             marginals_dict = {}

#         for col_name, col_size in zip(self.column_name, self.num_classes):
#             if col_name in marginals_dict:
#                 arr = marginals_dict[col_name]
#                 if len(arr) != col_size:
#                     raise ValueError('Invalid one-way marginal')
#             else:
#                 arr = np.ones(col_size, dtype=np.float32) / col_size

#             logits_list.append(torch.tensor(arr, device=self.device))

#         self.init_logits = torch.cat(logits_list, dim=0)

#         return self.init_logits

#     @torch.no_grad()
#     def uniform_sample(self, i=0):
#         '''
#         initialize a uniform distributed tensor x_T, as the start of the posterior process
#         '''
#         if (self.z[i] is None) or self.resample:
#             z_oh = []
#             for j in range(len(self.cum_num_classes)-1):
#                 start = self.cum_num_classes[j]
#                 end = self.cum_num_classes[j+1]

#                 probs = torch.tensor([1/(end-start) for _ in range(end-start)]).to(self.device)
#                 idxs = torch.multinomial(probs, self.batch_size, replacement=True)
#                 idxs = F.one_hot(idxs, num_classes=(end - start))
#                 idxs = torch.clamp(idxs, 1e-30, 1-(end-start)*1e-30)
#                 z_oh.append(idxs.float())

#             self.z[i] = torch.cat(z_oh, dim=1)

#         return self.z[i]
    
#     def predict_x(self, xt):
#         logits = self.model(xt)
#         data = []
#         for i in range(len(self.cum_num_classes)-1):
#             st = self.cum_num_classes[i]
#             ed = self.cum_num_classes[i+1]
#             temp_data = logits[:, st:ed].softmax(-1)

#             assert not torch.isnan(temp_data).any().item(), "invalid output (NaN)"
#             assert not torch.isinf(temp_data).any().item(), "invalid output (Inf)"
#             assert not (temp_data.sum(dim=-1) == 0).any().item(), 'invalid output (all zero)'
#             assert not (temp_data < 0).any().item(), 'invalid output (negative prob)'

#             data.append(temp_data)
#         return torch.cat(data, dim=1) 
#         # return logits.exp()
    

#     def compute_loss(self, i=0):
#         input = self.uniform_sample(i=i)
#         x_pred = self.predict_x(input)
#         if self.current_output is not None:
#             x_pred = x_pred + self.current_output

#         syn_attrs = [x_pred[:, q_id] for q_id in self.queries]
#         syn_answers = [syn_attr.prod(-1).mean(axis=0) for syn_attr in syn_attrs]
#         true_answers = [self.real_answers[i] for i in range(len(self.queries))]

#         return sum(self.query_weight[i] * (syn_answers[i] - true_answers[i])**2 for i in range(len(true_answers)))

#     def train_model(self, info, lr, iterations, output_loss = True, **kwargs):
#         for i in range(self.boost_num):
#             track = pd.DataFrame(columns=[f'loss'])
#             self.reset_model(i=i)
#             self.model.train()

#             self.optimizer = torch.optim.Adam(
#                     self.model.parameters(), 
#                     lr=lr
#                 )
#             # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, iterations, eta_min=1e-6)
#             for iter in range(iterations):
#                 self.optimizer.zero_grad()
#                 # self.anneal_enhace_weight(iter, iterations)

#                 loss = self.compute_loss(i)
#                 loss.backward()
#                 self.optimizer.step()
                
#                 # self.scheduler.step()
                
#                 if iter == 0:
#                     print('start loss:', loss.item())
#                 elif iter == iterations-1:
#                     print('end loss:', loss.item())

#                 track.loc[len(track)] = [loss.item()]

#             track.to_csv(os.path.join(self.parent_dir, f'{info}_loss_track_{i}.csv'))
#             torch.save(self.model.state_dict(), os.path.join(self.parent_dir, f'model_{i}.pt'))

#             with torch.no_grad():
#                 new_res = self.predict_x(self.uniform_sample(i=i))
#                 if self.current_output is None:
#                     self.current_output = new_res
#                 else:
#                     self.current_output += new_res

    
#     def sample(self, num_samples):
#         ind = torch.randint(0, self.current_output.shape[0], (num_samples,), dtype=torch.long)
#         res = self.current_output[ind]

#         return res


#     def map_to_marginal(self, x0_pred, marginal):
#         start_idx = [self.cum_num_classes[np.where(self.column_name == col)[0][0]] for col in marginal]
#         end_idx = [start + self.column_dims[col] for start, col in zip(start_idx, marginal)]

#         z_splits = [x0_pred[:, start:end] for start, end in zip(start_idx, end_idx)]

#         input_dims = ','.join(f'b{chr(105 + i)}' for i in range(len(z_splits)))
#         output_dims = ''.join(f'{chr(105 + i)}' for i in range(len(z_splits)))
#         einsum_str = f'{input_dims}->b{output_dims}'
#         joint_prob = torch.einsum(einsum_str, *z_splits).mean(dim=0).to(self.device)

#         return joint_prob.detach().cpu().numpy().flatten()

#     @torch.no_grad()
#     def obtain_sample_marginals(self, marginals, num_samples=1024):
#         x = self.sample(num_samples)
#         res = []
#         for marginal in marginals:
#             res.append(self.map_to_marginal(x, marginal))
        
#         return res


