'''
This conditional has been modified to achieve DP guarantee
'''


import numpy as np


class ConditionalGenerator(object):
    def __init__(self, data, domain, total_rho, **kwargs):
        self.model = []

        start = 0
        max_interval = 0
        counter = np.sum(int(x>1) for x in domain.values())
        max_interval = np.max(x for x in domain.values())

        rho = total_rho/counter

        self.interval = []
        self.n_col = 0
        self.n_opt = 0
        start = 0
        self.p = np.zeros((counter, max_interval))
        for item in domain.values():
            if item == 1:
                start += item
                continue
            elif item > 1:
                end = start + item

                tmp = np.sum(data[:, start:end], axis=0)
                tmp += np.sqrt(1/(2*rho)) * np.random.randn(*tmp.shape) # DP probability

                tmp = tmp / np.sum(tmp)
                self.p[self.n_col, :item] = tmp
                self.interval.append((self.n_opt, item))
                self.n_opt += item
                self.n_col += 1

                start = end
            else:
                assert 0

        self.interval = np.asarray(self.interval)

    def random_choice_prob_index(self, idx):
        a = self.p[idx]
        r = np.expand_dims(np.random.rand(a.shape[0]), axis=1)
        return (a.cumsum(axis=1) > r).argmax(axis=1)

    def sample(self, batch):
        if self.n_col == 0:
            return None

        batch = batch
        idx = np.random.choice(np.arange(self.n_col), batch)

        vec1 = np.zeros((batch, self.n_opt), dtype='float32')
        mask1 = np.zeros((batch, self.n_col), dtype='float32')
        mask1[np.arange(batch), idx] = 1
        opt1prime = self.random_choice_prob_index(idx)
        opt1 = self.interval[idx, 0] + opt1prime
        vec1[np.arange(batch), opt1] = 1

        return vec1, mask1, idx, opt1prime

    def sample_zero(self, batch):
        if self.n_col == 0:
            return None

        vec = np.zeros((batch, self.n_opt), dtype='float32')
        idx = np.random.choice(np.arange(self.n_col), batch)
        for i in range(batch):
            col = idx[i]
            # pick = int(np.random.choice(self.model[col]))
            pick = int(np.random.choice(np.arange(self.p.shape[1]), p=self.p[col,:]))
            vec[i, pick + self.interval[col, 0]] = 1

        return vec

    def generate_cond_from_condition_column_info(self, condition_info, batch):
        vec = np.zeros((batch, self.n_opt), dtype='float32')
        id = self.interval[condition_info["discrete_column_id"]][0] + condition_info["value_id"]
        vec[:, id] = 1
        return vec
