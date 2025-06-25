import torch
import torch.nn.functional as F
import numpy as np
import itertools
import copy
import os
import pandas as pd
from inspect import isfunction
from method.MargDL.scripts.denoiser import *
from method.MargDL.scripts.graph import *

def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, \
        f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)

    permute_order = (0, -1) + tuple(range(1, len(x.size())))

    x_onehot = x_onehot.permute(permute_order)

    log_x = torch.log(x_onehot.float().clamp(min=1e-30))

    return log_x


def prob_to_softmax_onehot(x, num_classes, tau=1.0):
    splits = torch.split(x, num_classes.tolist(), dim=1)
    one_hot_splits = [
        F.softmax(split, dim=1) for split in splits
    ]

    one_hot_x = torch.cat(one_hot_splits, dim=1)
    return one_hot_x 

def prob_to_logsoftmax_onehot(x, num_classes, tau=1.0):
    splits = torch.split(x, num_classes.tolist(), dim=1)
    one_hot_splits = [
        F.log_softmax(split, dim=1) for split in splits
    ]

    one_hot_x = torch.cat(one_hot_splits, dim=1)
    return one_hot_x




class QueryDiffusionG(nn.Module):
    def __init__(self, config, domain, device='cuda:0', **kwargs):
        super(QueryDiffusionG, self).__init__()
        self.config = config
        self.device = device
        self.parent_dir = kwargs.get('parent_dir', None)

        # diffusion networks
        self._denoise_fn = DenoiserModel(
            embedding_dim=self.config['model_params']['data_dim'],
            gen_dims=self.config['model_params']['d_layers'],
            data_dim=self.config['model_params']['data_dim']
        ).to(self.device)
        self._target_denoise_fn = copy.deepcopy(self._denoise_fn)
        self.num_timesteps = self.config['model_params']['num_timesteps']

        # data domain and marginals
        self.column_dims = domain
        self.num_classes = np.array(list(domain.values()))
        self.column_name = np.array(list(domain.keys()))
        self.cum_num_classes = np.concatenate((np.array([0]), np.cumsum(self.num_classes)))
        self.queries = None
        self.real_answers = None
        self.query_size = None
        self.query_weight = None
        self.jt = None
        self.Q_mask = None

        # sampling state
        self.batch_size = self.config['train']['batch_size']
        self.z = None
        self.resample = kwargs.get('resample', False)

        # precompute log noise schedule
        alphas = self.cosine_beta_schedule(self.num_timesteps)
        alphas = torch.tensor(alphas.astype('float64'))
        log_alpha = torch.log(alphas)
        log_cumprod_alpha = torch.cumsum(log_alpha, dim=0)
        log_1_min_alpha = self.log_1_min_a(log_alpha)
        log_1_min_cumprod_alpha = self.log_1_min_a(log_cumprod_alpha)

        self.register_buffer('log_alpha', log_alpha.float())
        self.register_buffer('log_1_min_alpha', log_1_min_alpha.float())
        self.register_buffer('log_cumprod_alpha', log_cumprod_alpha.float())
        self.register_buffer('log_1_min_cumprod_alpha', log_1_min_cumprod_alpha.float())

    def cosine_beta_schedule(self, timesteps, s=0.008):
        steps = timesteps + 1
        x = np.linspace(0, steps, steps)
        acp = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
        acp = acp / acp[0]
        alphas = (acp[1:] / acp[:-1])
        alphas = np.clip(alphas, a_min=0.001, a_max=1.)
        return np.sqrt(alphas)

    def log_1_min_a(self, a):
        return torch.log(1 - a.exp() + 1e-30)

    def merge_marginals(self, marginals):
        merged = {}
        for name, matrix, weight in marginals:
            if name not in merged:
                merged[name] = (matrix * weight, weight)
            else:
                ws, tw = merged[name]
                merged[name] = (ws + matrix * weight, tw + weight)
        result = []
        for name, (ws, tw) in merged.items():
            avg = ws / tw
            result.append((name, avg, tw))
        return result

    def find_query_index(self, marginals):
        index, answer, size, weight = [], [], [], []
        for marg, matrix, w in marginals:
            starts = [self.cum_num_classes[np.where(self.column_name == col)[0][0]] for col in marg]
            ends = [start + self.column_dims[col] for start, col in zip(starts, marg)]
            ranges = [range(s, e) for s, e in zip(starts, ends)]
            index += list(itertools.product(*ranges))
            answer += matrix.tolist()
            size += [1/matrix.size for _ in range(matrix.size)]
            weight += [w for _ in range(matrix.size)]
        ans = torch.tensor(answer, dtype=torch.float64, device=self.device)
        wgt = torch.tensor(weight, dtype=torch.float64, device=self.device)
        return index, ans, size, wgt

    def store_marginals(self, marginals, enhance_weight=1.0):
        merged = self.merge_marginals(marginals)
        cols = [m[0] for m in merged]
        self.jt = generate_junction_tree(list(self.column_dims.keys()), cols)
        self.queries, self.real_answers, self.query_size, self.query_weight = self.find_query_index(merged)
        K, D = len(self.queries), int(self.cum_num_classes[-1])
        self.Q_mask = torch.zeros((K, D), device=self.device)
        for k, q in enumerate(self.queries):
            self.Q_mask[k, q] = 1.0

    def initialize_logits(self, marginals=None):
        logits_list = []
        marg_dict = {m[0]: m[1] for m in marginals} if marginals else {}
        for col_name, col_size in zip(self.column_name, self.num_classes):
            if col_name in marg_dict:
                arr = marg_dict[col_name]
                if len(arr) != col_size:
                    raise ValueError('Invalid one-way marginal')
            else:
                arr = np.ones(col_size, dtype=np.float32) / col_size
            logits_list.append(torch.tensor(arr, device=self.device))
        self.init_logits = torch.cat(logits_list, dim=0)
        return self.init_logits

    @torch.no_grad()
    def uniform_sample(self):
        if self.z is None or self.resample:
            if not hasattr(self, 'init_logits'):
                raise RuntimeError('Please initialize sample logits')
            z_list = []
            for i in range(len(self.cum_num_classes)-1):
                start, end = self.cum_num_classes[i], self.cum_num_classes[i+1]
                probs = self.init_logits[start:end]
                idxs = torch.multinomial(probs, self.batch_size, replacement=True)
                oh = F.one_hot(idxs, num_classes=(end-start)).float().to(self.device)
                oh = torch.clamp(oh, 1e-30, 1 - (end-start)*1e-30)
                z_list.append(oh)
            self.z = torch.cat(z_list, dim=1)
        return self.z

    def predict_t(self, xt, t, use_target=False):
        model = self._denoise_fn if not use_target else self._target_denoise_fn
        logits = model(xt, t)
        parts = []
        for i in range(len(self.cum_num_classes)-1):
            st, ed = self.cum_num_classes[i], self.cum_num_classes[i+1]
            logp = logits[:, st:ed].log_softmax(-1)
            parts.append(torch.clamp(logp, min=-30, max=0.0))
        return torch.cat(parts, dim=1)

    def q_posterior_sample(self, xT, t, use_target=False):
        # This is used for generate input for next step
        xt = xT
        for i in reversed(range(t+1, self.num_timesteps)):
            time = torch.full((xT.size(0),), i, device=self.device, dtype=torch.float)
            xt = self.predict_t(xt, time, use_target).exp()
        return xt

    def posterior_answer(self, t):
        # vectorized posterior answers for all queries
        if t <= 0:
            return self.real_answers
        factor_a = self.log_cumprod_alpha[t-1].exp()
        factor_b = self.log_1_min_cumprod_alpha[t-1].exp()

        # query_size to tensor
        qsize = torch.tensor(self.query_size, dtype=torch.float64, device=self.device)
        a = factor_a * self.real_answers
        b = factor_b * qsize

        return a + b

    def compute_loss(self, t, use_target=True):
        with torch.no_grad():
            xT = self.uniform_sample()
            if t < self.num_timesteps - 1:
                xt1 = self.q_posterior_sample(xT, t, use_target)
                # xt1 = torch.clamp(xt1, 1e-30, 1 - xt1.size(1)*1e-30)
            else:
                xt1 = xT
        time = torch.full((self.batch_size,), t, device=self.device, dtype=torch.float)

        x_pred = self.predict_t(xt1, time, use_target=False) # generate this step without target model
        S = x_pred @ self.Q_mask.T
        syn = S.exp().mean(dim=0)
        real_ans = self.posterior_answer(t)
    
        if self.query_weight is not None:
            return (self.query_weight * (syn - real_ans)**2).sum()
        
        return ((syn - real_ans)**2).sum()

    def train_model(self, lr, iterations, save_loss=False, use_target=True, **kwargs):
        steps = self.num_timesteps
        if save_loss: track = pd.DataFrame(columns=[f'step {i+1}' for i in range(steps)])

        self._denoise_fn.train()
        self.optimizer = torch.optim.Adam(self._denoise_fn.parameters(), lr=lr)
        for iter in range(iterations):
            if use_target:
                self._target_denoise_fn.load_state_dict(self._denoise_fn.state_dict())
            losses = []
            for t in reversed(range(steps)):
                self.optimizer.zero_grad()
                loss = self.compute_loss(t, use_target)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())

            if save_loss: track.loc[len(track)] = losses[::-1]

        if save_loss: track.to_csv(os.path.join(self.parent_dir, f'loss_track.csv'))
        torch.save(self._denoise_fn.state_dict(), os.path.join(self.parent_dir, 'model.pt'))

    def sample(self, num_samples, return_logits=False):
        rounds = int(np.ceil(num_samples / self.batch_size))
        results = []
        for _ in range(rounds):
            xt = self.uniform_sample()
            for t in reversed(range(self.num_timesteps)):
                time = torch.full((self.batch_size,), t, device=self.device, dtype=torch.float)
                xt = self.predict_t(xt, time).exp()
            results.append(xt)
        samples = torch.cat(results, dim=0)
        idx = torch.randint(0, samples.shape[0], size=(num_samples,))
        samples = samples[idx, :]
        
        if return_logits:
            return samples 
        else:
            res = []
            for i in range(len(self.num_classes)):
                start, end = self.cum_num_classes[i], self.cum_num_classes[i + 1]
                probs = samples[:, start:end] 

                idxs = torch.multinomial(probs, num_samples=1).squeeze(dim=-1)  
                res.append(idxs)

            res = torch.stack(res, dim=1)
            return res.detach().cpu().numpy()

    @torch.no_grad()
    def obtain_sample_marginals(self, marginals, num_samples=1024):
        x0 = self.sample(num_samples, return_logits=True)
        return [self.map_to_marginal(x0, m) for m in marginals]

    def map_to_marginal(self, x_pred, marginal):
        start_idx = [self.cum_num_classes[np.where(self.column_name == col)[0][0]] for col in marginal]
        end_idx = [start + self.column_dims[col] for start, col in zip(start_idx, marginal)]

        z_splits = [x_pred[:, start:end] for start, end in zip(start_idx, end_idx)]

        input_dims = ','.join(f'b{chr(105 + i)}' for i in range(len(z_splits)))
        output_dims = ''.join(f'{chr(105 + i)}' for i in range(len(z_splits)))
        einsum_str = f'{input_dims}->b{output_dims}'
        joint_prob = torch.einsum(einsum_str, *z_splits).mean(dim=0).to(self.device)

        return joint_prob.detach().cpu().numpy().flatten()

    def test(self):
        l0 = self.compute_loss(0, use_target=False)
        lT = self.compute_loss(0, use_target=True)
        print('loss no target:', l0.item(), 'with target:', lT.item())

    def extract(self, a, t, x_shape):
        b = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(b, *([1] * (len(x_shape)-1)))

    def log_add_exp(self, a, b):
        maxv = torch.max(a, b)
        return maxv + torch.log(torch.exp(a - maxv) + torch.exp(b - maxv))





# class QueryDiffusion(torch.nn.Module):
#     def __init__(self, config, domain, device='cuda:0', **kwargs):
#         super(QueryDiffusion, self).__init__()
        
#         self.config = config
#         self.device = device
#         self.parent_dir = kwargs.get('parent_dir', None)

#         self._denoise_fn = DenoiserModel(
#                 embedding_dim=self.config['model_params']['data_dim'],
#                 gen_dims=self.config['model_params']['d_layers'],
#                 data_dim=self.config['model_params']['data_dim']
#             ).to(self.device)

#         self._target_denoise_fn = copy.deepcopy(self._denoise_fn)
#         self.num_timesteps = self.config['model_params']['num_timesteps']
#         self.column_dims = domain  # Dictionary storing column name -> one-hot dimension
#         self.num_classes = np.array(list(domain.values()))
#         self.column_name = np.array(list(domain.keys()))
#         self.marginals = []  # List to store (tuple a, array b, weight)

#         self.cum_num_classes = np.concatenate((np.array([0]), np.cumsum(self.num_classes)))

#         self.batch_size = self.config['train']['batch_size']
#         self.z = None
#         self.queries = None 
#         self.real_answers = None

#         alphas = self.cosine_beta_schedule(self.num_timesteps)
        
#         alphas = torch.tensor(alphas.astype('float64'))
#         log_alpha = np.log(alphas)
#         log_cumprod_alpha = np.cumsum(log_alpha)
        
#         log_1_min_alpha = self.log_1_min_a(log_alpha)
#         log_1_min_cumprod_alpha = self.log_1_min_a(log_cumprod_alpha)
        
#         self.register_buffer('log_alpha', log_alpha.float())
#         self.register_buffer('log_1_min_alpha', log_1_min_alpha.float())
#         self.register_buffer('log_cumprod_alpha', log_cumprod_alpha.float())
#         self.register_buffer('log_1_min_cumprod_alpha', log_1_min_cumprod_alpha.float())

        
#     def cosine_beta_schedule(self, timesteps, s=0.008):
#         steps = timesteps + 1
#         x = np.linspace(0, steps, steps)
#         alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
#         alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
#         alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
#         alphas = np.clip(alphas, a_min=0.001, a_max=1.)
#         return np.sqrt(alphas)
        
#     def log_1_min_a(self, a):
#         return torch.log(1 - a.exp() + 1e-30)
    
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

#     @torch.no_grad()
#     def uniform_sample(self):
#         '''
#         initialize a uniform distributed tensor x_T, as the start of the posterior process
#         '''
#         if self.z is None:
#             z_oh = []
#             for i in range(len(self.cum_num_classes)-1):
#                 start = self.cum_num_classes[i]
#                 end = self.cum_num_classes[i+1]

#                 probs = torch.tensor([1/(end-start) for _ in range(end-start)]).to(self.device)
#                 idxs = torch.multinomial(probs, self.batch_size, replacement=True)
#                 idxs = F.one_hot(idxs, num_classes=(end - start))
#                 idxs = torch.clamp(idxs, 1e-30, 1-(end-start)*1e-30)
#                 z_oh.append(idxs.float())

#             self.z = torch.cat(z_oh, dim=1).log()

#         return self.z
    
#     def predict_start(self, xt, t, use_target=False):
#         '''
#         This will return a softmax probability of prediction of xt-1
#         '''
#         model = self._denoise_fn if not use_target else self._target_denoise_fn
#         logits = model(xt, t)
#         data = []
#         for i in range(len(self.cum_num_classes)-1):
#             st = self.cum_num_classes[i]
#             ed = self.cum_num_classes[i+1]
#             data.append(logits[:, st:ed].log_softmax(-1))
#         return torch.cat(data, dim=1)


#     def q_posterior_sample(self, xT, t, use_target=False):
#         """
#         Given x_T (at t=T), iteratively denoise down to x_t.
#         will return a prob
#         """
#         xt = xT  # Start from x_T
#         for i in reversed(range(t+1, self.num_timesteps)):
#             time = torch.full((xT.size(0),), i, device=self.device, dtype=torch.float32)
#             x0_pred = self.predict_start(xt, time, use_target)
#             xt = self.q_posterior(x0_pred, xt, time)
#         return xt  # Return x_t
    
#     def q_pred(self, log_x_start, t):
#         log_cumprod_alpha_t = self.extract(self.log_cumprod_alpha, t.to(torch.int64), log_x_start.shape)
#         log_1_min_cumprod_alpha = self.extract(self.log_1_min_cumprod_alpha, t.to(torch.int64), log_x_start.shape)

#         return self.log_add_exp(
#             log_x_start + log_cumprod_alpha_t,
#             log_1_min_cumprod_alpha - np.log(np.sum(self.num_classes))
#         )

#     def q_pred_one_timestep(self, log_x_t, t):
#         log_alpha_t = self.extract(self.log_alpha, t.to(torch.int64), log_x_t.shape)
#         log_1_min_alpha_t = self.extract(self.log_1_min_alpha, t.to(torch.int64), log_x_t.shape)

#         # alpha_t * E[xt] + (1 - alpha_t) 1 / K
#         splits = torch.split(log_x_t, self.num_classes.tolist(), dim=1)

#         log_probs = torch.cat([
#             self.log_add_exp(split + log_alpha_t.expand(-1, split.shape[1]), 
#                             log_1_min_alpha_t.expand(-1, split.shape[1]) - np.log(self.num_classes[i]))
#             for i, split in enumerate(splits)
#         ], dim=1)

#         return log_probs

#     def q_posterior(self, log_x_start, log_x_t, t):
#         # the input and output are log_prob
#         # q(xt-1 | xt, x0) = q(xt | xt-1, x0) * q(xt-1 | x0) / q(xt | x0)
#         # where q(xt | xt-1, x0) = q(xt | xt-1).

#         t_minus_1 = t - 1
#         # Remove negative values, will not be used anyway for final decoder
#         t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
#         log_EV_qxtmin_x0 = self.q_pred(log_x_start, t_minus_1)

#         num_axes = (1,) * (len(log_x_start.size()) - 1)
#         t_broadcast = t.view(-1, *num_axes) * torch.ones_like(log_x_start)
#         log_EV_qxtmin_x0 = torch.where(t_broadcast == 0, log_x_start, log_EV_qxtmin_x0)

#         # Note: _NOT_ x_tmin1, which is how the formula is typically used!!!
#         # Not very easy to see why this is true. But it is :)
#         unnormed_logprobs = log_EV_qxtmin_x0 + self.q_pred_one_timestep(log_x_t, t)

#         log_EV_xtmin_given_xt_given_xstart = \
#             unnormed_logprobs \
#             - torch.logsumexp(unnormed_logprobs, dim=1, keepdim=True)

#         return log_EV_xtmin_given_xt_given_xstart
    
#     def map_to_marginal(self, x0_pred, marginal):
#         start_idx = [self.cum_num_classes[np.where(self.column_name == col)[0][0]] for col in marginal]
#         end_idx = [start + self.column_dims[col] for start, col in zip(start_idx, marginal)]

#         one_hot_x0_pred = prob_to_softmax_onehot(x0_pred, self.num_classes)
#         z_splits = [one_hot_x0_pred[:, start:end] for start, end in zip(start_idx, end_idx)]
#         # z_splits = [x0_pred[:, start:end] for start, end in zip(start_idx, end_idx)]

#         input_dims = ','.join(f'b{chr(105 + i)}' for i in range(len(z_splits)))
#         output_dims = ''.join(f'{chr(105 + i)}' for i in range(len(z_splits)))
#         einsum_str = f'{input_dims}->b{output_dims}'
#         joint_prob = torch.einsum(einsum_str, *z_splits).mean(dim=0).to(self.device)

#         return joint_prob
    

#     def posterior_answer(self, query_id, t):
#         if t <= 0:
#             return self.real_answers[query_id]
#         else:
#             # indices = [idx.item() for idx in list(self.queries[query_id])]
#             return self.log_cumprod_alpha[t-1].exp() * self.real_answers[query_id] +\
#                   self.log_1_min_cumprod_alpha[t-1].exp() / self.query_size[query_id]

#     def update_target_denoiser(self):
#         self._target_denoise_fn.load_state_dict(self._denoise_fn.state_dict())


#     def compute_loss(self, t, use_target=True):
#         with torch.no_grad():
#             xT = self.uniform_sample()
#             if t < self.num_timesteps - 1:
#                 xt_plus_1 = self.q_posterior_sample(xT, t, use_target=use_target)
#             else:
#                 xt_plus_1 = xT

#         x0_pred = self.predict_start(xt_plus_1, torch.full((self.batch_size,), t, device=self.device, dtype=torch.float))
#         xt_pred = self.q_posterior(x0_pred, xt_plus_1, torch.full((self.batch_size,), t, device=self.device, dtype=torch.float)).exp()

#         syn_attrs = [xt_pred[:, q_id] for q_id in self.queries]
#         syn_answers = [syn_attr.prod(-1).mean(axis=0) for syn_attr in syn_attrs]
#         true_answers = [self.posterior_answer(i, t) for i in range(len(self.queries))]

#         return sum((syn_answers[i] - true_answers[i]).abs() for i in range(len(true_answers)))


#     def train_model(self, info, lr, iterations, use_target):
#         steps = self.config['model_params']['num_timesteps']
#         track = pd.DataFrame(columns=[f'step {i+1}' for i in range(steps)])

#         self._denoise_fn.train()

#         print(f'----------------------------- {info} loss track -------------------------------')
#         print("iter   " + "  ".join([f"step {i+1} " for i in range(steps)]))

#         self.optimizer = torch.optim.Adam(
#                 self._denoise_fn.parameters(), 
#                 lr=lr
#             )
#         self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, iterations, eta_min=1e-8)
#         for iter in range(iterations):
#             loss_list = [0.0 for _ in range(steps)]
#             if use_target: 
#                 self.update_target_denoiser()

#             for t in reversed(range(steps)):
#                 self.optimizer.zero_grad()
#                 loss = self.compute_loss(t, use_target=use_target)

#                 loss_list[t] = loss.item()
#                 loss.backward()
#                 self.optimizer.step()
            
#             self.scheduler.step()
#             print(f"{iter:<5}  " + "  ".join(f"{v:.4f}" for v in loss_list))
#             track.loc[len(track)] = loss_list

#         print('\n')
#         track.to_csv(os.path.join(self.parent_dir, f'{info}_loss_track.csv'))
#         torch.save(self._denoise_fn.state_dict(), os.path.join(self.parent_dir, 'model.pt'))

#         self.test()
    
#     def test(self):
#         loss = self.compute_loss(0, use_target=False)
#         print('No target loss:', loss.item()) 

#         loss = self.compute_loss(0, use_target=True)
#         print('target loss:', loss.item()) 

#     def sample(self, num_samples):
#         # self._denoise_fn.load_state_dict(torch.load(os.path.join(self.parent_dir, 'model.pt')))
#         # self._denoise_fn.eval()

#         rounds = int(np.ceil(num_samples/self.batch_size))
#         res = None

#         for i in range(rounds):
#             xt = self.uniform_sample()
#             for t in reversed(range(self.num_timesteps)):
#                 x0_pred = self.predict_start(xt, torch.full((self.batch_size,), t, device=self.device, dtype=torch.float))
#                 xt = self.q_posterior(x0_pred, xt, torch.full((self.batch_size,), t, device=self.device, dtype=torch.float))

#                 if i == 0:
#                     self.test()
#                     # raise ValueError('debug mode')
#                     if t == 0:
#                         syn_attrs = [xt[:, q_id] for q_id in self.queries]
#                         syn_answers = [syn_attr.prod(-1).mean(axis=0) for syn_attr in syn_attrs]
#                         true_answers = [self.posterior_answer(i, t) for i in range(len(self.queries))]

#                         print('syn error', sum((syn_answers[i] - true_answers[i]).abs() for i in range(len(true_answers))))


#             if res is None:
#                 res = xt 
#             else:
#                 res = torch.cat((res, xt), dim=0)

#         return res.exp()
    

#     @torch.no_grad()
#     def obtain_sample_marginals(self, marginals, num_samples=1024):
#         x0 = self.sample(num_samples)
#         res = []
#         for marginal in marginals:
#             res.append(self.map_to_marginal(x0, marginal))
        
#         return res

#     def extract(self, a, t, x_shape):
#         b, *_ = t.shape
#         out = a.gather(-1, t)
#         return out.reshape(b, *((1,) * (len(x_shape) - 1)))
    
#     def log_add_exp(self, a, b):
#         maximum = torch.max(a, b)
#         return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))
    




# class MargDiffusion(torch.nn.Module):
#     def __init__(self, config, domain, device='cuda:0', **kwargs):
#         super(MargDiffusion, self).__init__()
        
#         self.config = config
#         self.device = device
#         self.parent_dir = kwargs.get('parent_dir', None)

#         self.num_timesteps = self.config['model_params']['num_timesteps']
#         self.column_dims = domain  # Dictionary storing column name -> one-hot dimension
#         self.num_classes = np.array(list(domain.values()))
#         self.column_name = np.array(list(domain.keys()))
#         self.marginals = []  # List to store (tuple a, array b, weight)

#         self.cum_num_classes = np.concatenate((np.array([0]), np.cumsum(self.num_classes)))

#         self.batch_size = self.config['train']['batch_size']
#         self.resample = kwargs.get('parent_dir', False)
#         self.latent_z = [None for i in range(self.num_timesteps)]

#         alphas = self.cosine_beta_schedule(self.num_timesteps)
        
#         alphas = torch.tensor(alphas.astype('float64'))
#         log_alpha = np.log(alphas)
#         log_cumprod_alpha = np.cumsum(log_alpha)
        
#         log_1_min_alpha = self.log_1_min_a(log_alpha)
#         log_1_min_cumprod_alpha = self.log_1_min_a(log_cumprod_alpha)
        
#         self.register_buffer('log_alpha', log_alpha.float())
#         self.register_buffer('log_1_min_alpha', log_1_min_alpha.float())
#         self.register_buffer('log_cumprod_alpha', log_cumprod_alpha.float())
#         self.register_buffer('log_1_min_cumprod_alpha', log_1_min_cumprod_alpha.float())

#     def reset_denoiser(self):
#         self._denoise_fn = Generator(
#                 embedding_dim=self.config['model_params']['data_dim'],
#                 gen_dims=self.config['model_params']['d_layers'],
#                 data_dim=self.config['model_params']['data_dim']
#             ).to(self.device)
        
#     def cosine_beta_schedule(self, timesteps, s=0.008):
#         steps = timesteps + 1
#         x = np.linspace(0, steps, steps)
#         alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
#         alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
#         alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
#         alphas = np.clip(alphas, a_min=0.001, a_max=1.)
#         return np.sqrt(alphas)
        
#     def log_1_min_a(self, a):
#         return torch.log(1 - a.exp() + 1e-30)
    
#     def find_query_index(self, marginals):
#         index = []
#         answer = []
#         size = []
#         for (marg, matrix, _) in marginals:
#             start_idx = [self.cum_num_classes[np.where(self.column_name == col)[0][0]] for col in marg]
#             end_idx = [start + self.column_dims[col] for start, col in zip(start_idx, marg)]
#             iter_list = [range(a,b) for (a, b) in zip(start_idx, end_idx)]
#             index += list(itertools.product(*iter_list))
#             answer += matrix.flatten().tolist()
#         return index, torch.tensor(answer, device=self.device)


#     def store_marginals(self, marginals):
#         '''
#         tansfer marginal to queries and store them
#         '''
#         self.marginal_list = [marg_list[0]for marg_list in marginals]
#         self.queries, self.real_answers = self.find_query_index(marginals)


#     def initialize_logits(self, marginals=None):
#         logits_list = []
#         self.init_marginals = marginals 

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
#     def uniform_sample(self):
#         '''
#         initialize a tensor x_T as the start of the posterior process
#         '''
#         if self.latent_z[-1] is None or self.resample:
#             if self.resample:
#                 self.initialize_logits(self.init_marginals)

#             z_oh = []
#             for i in range(len(self.cum_num_classes)-1):
#                 start = self.cum_num_classes[i]
#                 end = self.cum_num_classes[i+1]

#                 probs = self.init_logits[start: end]
#                 idxs = torch.multinomial(probs, self.batch_size, replacement=True)
#                 z_oh.append(F.one_hot(idxs, num_classes=(end - start)).float())

#             self.latent_z[-1] = torch.cat(z_oh, dim=1)

#         return self.latent_z[-1]
    
#     def predict_t(self, xt, t, model_type = 'load'):
#         '''
#         This will return a softmax probability of prediction of xt-1
#         '''
#         if model_type == 'load':
#             model = copy.deepcopy(self._denoise_fn).to(self.device)
#             model.load_state_dict(torch.load(os.path.join(self.parent_dir, f'model_{t}.pt')))
#             logits = model(xt)
#         else:
#             logits = self._denoise_fn(xt)
        

#         data = []
#         for i in range(len(self.cum_num_classes)-1):
#             st = self.cum_num_classes[i]
#             ed = self.cum_num_classes[i+1]
#             data.append(logits[:, st:ed].softmax(-1))
#         return torch.cat(data, dim=1)


#     def q_posterior_sample(self, xT, t):
#         """
#         Given x_T (at t=T), iteratively denoise down to x_t.
#         will return a prob
#         """
#         if self.resample:
#             xt = xT  # Start from x_T
#             for i in reversed(range(t, self.num_timesteps)):
#                 xt = self.predict_t(xt, i, model_type = 'load')
#         else:
#             xt = self.latent_z[t]
#         return xt  # Return x_t
    
    
#     def map_to_marginal(self, x0_pred, marginal):
#         start_idx = [self.cum_num_classes[np.where(self.column_name == col)[0][0]] for col in marginal]
#         end_idx = [start + self.column_dims[col] for start, col in zip(start_idx, marginal)]

#         one_hot_x0_pred = prob_to_softmax_onehot(x0_pred, self.num_classes)
#         z_splits = [one_hot_x0_pred[:, start:end] for start, end in zip(start_idx, end_idx)]
#         # z_splits = [x0_pred[:, start:end] for start, end in zip(start_idx, end_idx)]

#         input_dims = ','.join(f'b{chr(105 + i)}' for i in range(len(z_splits)))
#         output_dims = ''.join(f'{chr(105 + i)}' for i in range(len(z_splits)))
#         einsum_str = f'{input_dims}->b{output_dims}'
#         joint_prob = torch.einsum(einsum_str, *z_splits).mean(dim=0).to(self.device)

#         return joint_prob
    
#     @torch.no_grad()
#     def posterior_answer(self, query_id, t):
#         if t <= 0:
#             # print('use original answer')
#             return self.real_answers[query_id]
#         else:
#             return self.log_cumprod_alpha[t-1].exp() * self.real_answers[query_id] +\
#                   self.log_1_min_cumprod_alpha[t-1].exp() * torch.prod(self.init_logits[list(self.queries[query_id])])

#     def compute_loss(self, t, output=False):
#         with torch.no_grad():
#             xT = self.uniform_sample()
#             if t < self.num_timesteps - 1:
#                 xt_plus_1 = self.q_posterior_sample(xT, t+1)
#                 xt_plus_1 = prob_to_softmax_onehot(xt_plus_1, self.num_classes)
#             else:
#                 xt_plus_1 = xT

#         xt_pred = self.predict_t(xt_plus_1, t, model_type = 'train') # note that this is one-hot

#         syn_attrs = [xt_pred[:, q_id] for q_id in self.queries]
#         syn_answers = [syn_attr.prod(-1).mean(axis=0) for syn_attr in syn_attrs]
#         # true_answers = [self.posterior_answer(i, t) for i in range(len(self.queries))]
#         true_answers = self.real_answers

#         if output:
#             print('target marg:', true_answers)

#         return sum((syn_answers[i] - true_answers[i]).abs() for i in range(len(true_answers)))

#     def train_model(self, info, lr, iterations, **kwargs):
#         for t in reversed(range(self.num_timesteps)):
#             print(t)
#             track = pd.DataFrame(columns=['iter', f'step {t} loss'])

#             self.reset_denoiser()
#             self._denoise_fn.train()
#             self.optimizer = torch.optim.Adam(
#                     self._denoise_fn.parameters(), 
#                     lr=lr
#                 )
#             self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, iterations, eta_min=1e-8)

#             print(f'----------------------------- {info} loss track, step {t} -------------------------------')
#             print("iter   loss")
#             for iter in range(iterations):

#                 self.optimizer.zero_grad()
#                 loss = self.compute_loss(t)

#                 loss.backward()
#                 self.optimizer.step()
#                 self.scheduler.step()

#                 if iter == 0 or iter%10 == 9:
#                     print(f"{iter:<5}  {loss.item():.4f}")
#                 track.loc[len(track)] = [iter, loss.item()]

#             self.save_model(t)
#             track.to_csv(os.path.join(self.parent_dir, f'{info}_{t}_loss_track.csv'))
#             print('\n')


#     def update_latent_z(self, t):
#         xT = self.uniform_sample()
#         xt = self.q_posterior_sample(xT, t)
#         self.latent_z[t-1] = xt


#     def save_model(self, t):
#         torch.save(self._denoise_fn.state_dict(), os.path.join(self.parent_dir, f'model_{t}.pt'))
    

#     @torch.no_grad()
#     def sample(self, num_samples):
#         rounds = int(np.ceil(num_samples/self.batch_size))
#         res = None

#         for i in range(rounds):
#             xt = self.uniform_sample()
#             for t in reversed(range(self.num_timesteps)):
#                 xt = self.predict_t(xt, t, model_type='load')
            
#             if res is None:
#                 res = xt 
#             else:
#                 res = torch.cat((res, xt), dim=0)
        
#         res = res[:num_samples, :]
#         return res
    

#     @torch.no_grad()
#     def obtain_sample_marginals(self, marginals, num_samples=1024):
#         x0 = self.sample(num_samples)
#         res = []
#         for marginal in marginals:
#             res.append(self.map_to_marginal(x0, marginal))
        
#         return res

#     def extract(self, a, t, x_shape):
#         b, *_ = t.shape
#         out = a.gather(-1, t)
#         return out.reshape(b, *((1,) * (len(x_shape) - 1)))
    
#     def log_add_exp(self, a, b):
#         maximum = torch.max(a, b)
#         return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))








# class MarginalDiffusion(torch.nn.Module):
#     def __init__(self, denoise_fn, domain, timesteps=100, device='cuda:0'):
#         super(MarginalDiffusion, self).__init__()
        
#         self.device = device
#         self._denoise_fn = denoise_fn
#         self.num_timesteps = timesteps
#         self.column_dims = domain  # Dictionary storing column name -> one-hot dimension
#         self.num_classes = np.array(list(domain.values()))
#         self.column_name = np.array(list(domain.keys()))
#         self.marginals = []  # List to store (tuple a, array b, weight)

#         self.cum_num_classes = np.concatenate((np.array([0]), np.cumsum(self.num_classes)))
        
#         alphas = self.cosine_beta_schedule(timesteps)
        
#         alphas = torch.tensor(alphas.astype('float32'))
#         log_alpha = np.log(alphas)
#         log_cumprod_alpha = np.cumsum(log_alpha)
        
#         log_1_min_alpha = self.log_1_min_a(log_alpha)
#         log_1_min_cumprod_alpha = self.log_1_min_a(log_cumprod_alpha)
        
#         self.register_buffer('log_alpha', log_alpha.float())
#         self.register_buffer('log_1_min_alpha', log_1_min_alpha.float())
#         self.register_buffer('log_cumprod_alpha', log_cumprod_alpha.float())
#         self.register_buffer('log_1_min_cumprod_alpha', log_1_min_cumprod_alpha.float())
        
#     def cosine_beta_schedule(self, timesteps, s=0.008):
#         steps = timesteps + 1
#         x = np.linspace(0, steps, steps)
#         alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
#         alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
#         alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
#         alphas = np.clip(alphas, a_min=0.001, a_max=1.)
#         return np.sqrt(alphas)
        
#     def log_1_min_a(self, a):
#         return torch.log(1 - a.exp() + 1e-30)

#     def store_marginals(self, marginals):
#         self.marginals = marginals
#         self.real_marginals = [marg_list[1].to(self.device) for marg_list in self.marginals]
#         self.marginal_list = [marg_list[0]for marg_list in self.marginals]
    
    
#     @torch.no_grad()
#     def uniform_sample(self, batch_size):
#         '''
#         initialize a uniform distributed tensor x_T, as the start of the posterior process
#         '''
#         indices = [
#             torch.randint(0, num_class, (batch_size,), device=self.device)
#             for num_class in self.num_classes
#         ]

#         one_hot_splits = [
#             F.one_hot(idx, num_classes=num_class).float() * (1 - 1e-30) + 1e-30 / num_class
#             for idx, num_class in zip(indices, self.num_classes)
#         ]

#         log_z = torch.cat(one_hot_splits, dim=1).log()
#         return log_z
    
#     def predict_start(self, xt, t):
#         '''
#         This will return a softmax probability of prediction of x0
#         This need to be transformed into log probability if it is used to calculate xt-1
#         '''
#         logits = self._denoise_fn(xt, t)
#         splits = torch.split(logits, self.num_classes.tolist(), dim=1)
#         x0_pred = torch.cat([F.softmax(split, dim=1) for split in splits], dim=1)
#         return x0_pred


#     def q_posterior_sample(self, xT, t):
#         """
#         Given x_T (at t=T), iteratively denoise down to x_t.
#         """
#         xt = xT  # Start from x_T
#         with torch.no_grad():
#             for i in reversed(range(t+1, self.num_timesteps)):
#                 time = torch.full((xT.size(0),), i, device=self.device, dtype=torch.float)
#                 x0_pred = self.predict_start(xt, time).log()
#                 xt = self.q_posterior(x0_pred, xt, time)
#         return xt  # Return x_t
    
#     def q_pred_one_timestep(self, log_x_t, t):
#         log_alpha_t = self.extract(self.log_alpha, t, log_x_t.shape)
#         log_1_min_alpha_t = self.extract(self.log_1_min_alpha, t, log_x_t.shape)

#         # alpha_t * E[xt] + (1 - alpha_t) 1 / K
#         splits = torch.split(log_x_t, self.num_classes.tolist(), dim=1)

#         log_probs = torch.cat([
#             self.log_add_exp(split + log_alpha_t.expand(-1, split.shape[1]), 
#                             log_1_min_alpha_t.expand(-1, split.shape[1]) - np.log(self.num_classes[i]))
#             for i, split in enumerate(splits)
#         ], dim=1)

#         return log_probs
    
#     def q_pred(self, log_x_start, t):
#         log_cumprod_alpha_t = self.extract(self.log_cumprod_alpha, t, log_x_start.shape)
#         log_1_min_cumprod_alpha = self.extract(self.log_1_min_cumprod_alpha, t, log_x_start.shape)

#         return self.log_add_exp(
#             log_x_start + log_cumprod_alpha_t,
#             log_1_min_cumprod_alpha - np.log(np.sum(self.num_classes))
#         )

#     def q_posterior(self, log_x_start, log_x_t, t):
#         # q(xt-1 | xt, x0) = q(xt | xt-1, x0) * q(xt-1 | x0) / q(xt | x0)
#         # where q(xt | xt-1, x0) = q(xt | xt-1).

#         t_minus_1 = t - 1
#         # Remove negative values, will not be used anyway for final decoder
#         t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
#         log_EV_qxtmin_x0 = self.q_pred(log_x_start, t_minus_1)

#         num_axes = (1,) * (len(log_x_start.size()) - 1)
#         t_broadcast = t.view(-1, *num_axes) * torch.ones_like(log_x_start)
#         log_EV_qxtmin_x0 = torch.where(t_broadcast == 0, log_x_start, log_EV_qxtmin_x0)

#         # Note: _NOT_ x_tmin1, which is how the formula is typically used!!!
#         # Not very easy to see why this is true. But it is :)
#         unnormed_logprobs = log_EV_qxtmin_x0 + self.q_pred_one_timestep(log_x_t, t)

#         log_EV_xtmin_given_xt_given_xstart = \
#             unnormed_logprobs \
#             - torch.logsumexp(unnormed_logprobs, dim=1, keepdim=True)

#         return log_EV_xtmin_given_xt_given_xstart
    

#     def map_to_marginal(self, xt, marginal):
#         start_idx = [self.cum_num_classes[np.where(self.column_name == col)[0][0]] for col in marginal]
#         end_idx = [start + self.column_dims[col] for start, col in zip(start_idx, marginal)]

#         z_splits = [xt[:, start:end] for start, end in zip(start_idx, end_idx)]
#         # z_splits = [x0_pred[:, start:end] for start, end in zip(start_idx, end_idx)]

#         input_dims = ','.join(f'b{chr(105 + i)}' for i in range(len(z_splits)))
#         output_dims = ''.join(f'{chr(105 + i)}' for i in range(len(z_splits)))
#         einsum_str = f'{input_dims}->b{output_dims}'
#         joint_prob = torch.einsum(einsum_str, *z_splits).mean(dim=0).to(self.device)

#         return joint_prob

    
#     def posterior_marginal(self, final_marginal, t):
#         if t <= 1:
#             return final_marginal.to(self.device)
#         else:
#             marg = self.log_cumprod_alpha[t-1] * final_marginal.to(self.device) +\
#                   self.log_1_min_cumprod_alpha[t-1] * torch.full(final_marginal.size(), 1/final_marginal.numel(), dtype=torch.float, device=self.device)
#         return marg

#     def compute_distribution_loss(self, mapped_dist, target_dist):
#         # print('syn:', mapped_dist)
#         # print('real:', target_dist)
#         return torch.norm(mapped_dist - target_dist.to(self.device), p=2)
#         # return torch.kl_div(mapped_dist, target_dist.to(self.device), reduction='batchmean')


#     def compute_loss(self, batch_size, t):
#         xT = self.uniform_sample(batch_size)
#         if t < self.num_timesteps - 1:
#             xt_plus_1 = self.q_posterior_sample(xT, t)
#             xt_plus_1 = log_prob_to_softmax_onehot(xt_plus_1, self.num_classes)
#         else:
#             xt_plus_1 = xT

#         x0_pred = self.predict_start(xt_plus_1, torch.full((batch_size,), t, device=self.device, dtype=torch.float))
#         if t > 0:
#             time = torch.full((batch_size,), t, device=self.device, dtype=torch.float)
#             xt = log_prob_to_softmax_onehot(self.q_posterior(x0_pred.log(), xt_plus_1, time),  self.num_classes)
#         else:
#             xt = x0_pred

#         total_loss = 0
#         for (marginal, target_marg, weight) in self.marginals:
#             mapped_marg = self.map_to_marginal(xt, marginal)
#             loss = self.compute_distribution_loss(mapped_marg, self.posterior_marginal(target_marg, t)) * weight
#             total_loss += loss
#         return total_loss

#     @torch.no_grad()
#     def sample(self, num_samples, sample_batch=4096):
#         rounds = int(np.ceil(num_samples/sample_batch))
#         res = None

#         for i in range(rounds):
#             num_sample = sample_batch if i < rounds-1 else num_samples - (rounds-1)*sample_batch

#             xt = self.uniform_sample(num_sample)
#             for t in reversed(range(self.num_timesteps)):
#                 time = torch.full((num_sample,), t, device=self.device, dtype=torch.float)
#                 x0_pred = self.predict_start(xt, time)
#                 if t > 0:
#                     xt = self.q_posterior(x0_pred.log(), xt, time)
#                     xt = log_prob_to_logsoftmax_onehot(xt, self.num_classes)          
#                 else:
#                     xt = x0_pred
                
#             if res is None:
#                 res = xt 
#             else:
#                 res = torch.cat((res, xt), dim=0)
#         return res

#     @torch.no_grad()
#     def obtain_sample_marginals(self, marginals, num_samples=4096):
#         x0 = self.sample(num_samples)
#         res = []
#         for marginal in marginals:
#             res.append(self.map_to_marginal(x0, marginal))
        
#         return res
    
#     @torch.no_grad()
#     def find_max_error_query(self, num_samples=4096):
#         syn_marginals = self.obtain_sample_marginals([marg_list[0] for marg_list in self.marginals], num_samples=num_samples)
#         gap_info = (
#             (i, np.unravel_index((torch.abs(syn - real)).view(-1).argmax().item(), syn.shape),
#             torch.abs(syn - real).view(-1).max().item())
#             for i, (syn, real) in enumerate(zip(syn_marginals, self.real_marginals))
#         )
#         max_error = max(gap_info, key=lambda x: x[2])
#         return max_error[0], max_error[1], self.real_marginals[max_error[0]][max_error[1]]


#     def extract(self, a, t, x_shape):
#         b, *_ = t.shape
#         out = a.gather(-1, t)
#         return out.reshape(b, *((1,) * (len(x_shape) - 1)))
    
#     def log_add_exp(self, a, b):
#         maximum = torch.max(a, b)
#         return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


