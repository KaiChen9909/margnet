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
