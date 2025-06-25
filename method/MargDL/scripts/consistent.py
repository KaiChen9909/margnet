import numpy as np
import copy


def norm_sub(count):
    summation = np.sum(count)
    lower = 0.0
    upper = -np.sum(count[count < 0.0])
    current = 0.0
    delta = 0.0
    while abs(summation - current) > 1.0:
        delta = (lower + upper) / 2.0
        new_count = count - delta
        new_count[new_count < 0.0] = 0.0
        current = np.sum(new_count)
        if current < summation:
            upper = delta
        else:
            lower = delta
    count = count - delta
    count[count < 0.0] = 0.0
    return count


def norm_cut(count):
    negative = np.where(count < 0.0)[0]
    neg_total = abs(np.sum(count[negative]))
    count[negative] = 0.0
    positive = np.where(count > 0.0)[0]
    if positive.size > 0:
        sorted_pos = positive[np.argsort(count[positive])]
        cumsum = np.cumsum(count[sorted_pos])
        thresh = np.where(cumsum <= neg_total)[0]
        if thresh.size == 0:
            i = sorted_pos[0]
            count[i] = cumsum[0] - neg_total
        else:
            count[sorted_pos[thresh]] = 0.0
            next_i = thresh[-1] + 1
            if next_i < sorted_pos.size:
                i = sorted_pos[next_i]
                count[i] = cumsum[next_i] - neg_total
    else:
        count[:] = 0.0
    return count


class ConsistencyTuple:
    class Marg:
        def __init__(self, name_tuple, count_array, weight, attr_index, num_categories):
            self.name_tuple = name_tuple
            self.attr_set = set(name_tuple)
            self.attributes_index = [attr_index[attr] for attr in name_tuple]
            self.num_categories = np.array(num_categories)
            self.num_attributes = len(num_categories)
            self.ways = len(self.attributes_index)
            self.num_key = int(np.prod(self.num_categories[self.attributes_index])) if self.ways > 0 else 1
            self.count = count_array.copy()
            # rho = sqrt(1/weight)
            self.rho = np.sqrt(1.0 / weight) if weight > 0 else 0.0
            self.encode_num = np.zeros(self.ways, dtype=np.uint32)
            self.cum_mul = np.zeros(self.ways, dtype=np.uint32)
            self.calculate_encode_num()
            self.calculate_tuple_key()
            self.get_sum()

        def calculate_encode_num(self):
            if self.ways > 0:
                idx = np.array(self.attributes_index, dtype=int)
                dims = self.num_categories[idx]
                tmp = np.roll(dims, 1)
                tmp[0] = 1
                self.cum_mul = np.cumprod(tmp)
                dims2 = self.num_categories[idx]
                tmp2 = np.roll(dims2, self.ways - 1)
                tmp2[-1] = 1
                tmp2 = tmp2[::-1]
                self.encode_num = np.cumprod(tmp2)[::-1]
            else:
                self.encode_num = np.zeros(1, dtype=np.uint32)
                self.cum_mul = np.zeros(1, dtype=np.uint32)

        def calculate_tuple_key(self):
            if self.ways > 0:
                self.tuple_key = np.zeros((self.num_key, self.ways), dtype=np.uint32)
                for i, ai in enumerate(self.attributes_index):
                    cats = np.arange(self.num_categories[ai])
                    col = np.tile(np.repeat(cats, self.encode_num[i]), self.cum_mul[i])
                    self.tuple_key[:, i] = col
            else:
                self.tuple_key = np.array([[0]], dtype=np.uint32)

        def get_sum(self):
            self.sum = np.sum(self.count)

        def init_consist_parameters(self, n):
            self.summations = np.zeros((self.num_key, n))
            self.weights = np.zeros(n)
            self.rhos = np.zeros(n)

        def project_from_bigger(self, bigger, index):
            full_enc = np.zeros(self.num_attributes, dtype=np.uint32)
            for i, ai in enumerate(self.attributes_index):
                full_enc[ai] = self.encode_num[i]
            enc_used = full_enc[bigger.attributes_index]
            code = np.matmul(bigger.tuple_key, enc_used)
            diff = np.setdiff1d(bigger.attributes_index, self.attributes_index)
            self.weights[index] = 1.0 / np.product(self.num_categories[diff]) if diff.size > 0 else 1.0
            self.rhos[index] = bigger.rho
            self.summations[:, index] = np.bincount(code, weights=bigger.count, minlength=self.num_key)

        def calculate_delta(self):
            w = self.rhos * self.weights
            target = np.matmul(self.summations, w) / np.sum(w)
            self.delta = - (self.summations.T - target).T * w

        def update_marg(self, common, index):
            full_enc = np.zeros(self.num_attributes, dtype=np.uint32)
            for i, ai in enumerate(common.attributes_index):
                full_enc[ai] = common.encode_num[i]
            enc_used = full_enc[self.attributes_index]
            code = np.matmul(self.tuple_key, enc_used)
            si = np.argsort(code)
            _, cnt = np.unique(code, return_counts=True)
            np.add.at(self.count, si, np.repeat(common.delta[:, index], cnt))

        def non_negativity(self, method):
            if method == 'N1':
                self.count = norm_cut(self.count)
            elif method == 'N2':
                self.count = norm_sub(self.count)
            else:
                raise ValueError('invalid non_negativity')
            self.get_sum()

    def __init__(self, marginal_list, domain_attrs, domain_shape, consist_iterations=10, non_negativity='N1'):
        self.iterations = consist_iterations
        self.non_negativity_method = non_negativity
        self.domain_attrs = list(domain_attrs)
        self.domain_shape = domain_shape
        self.attr_index = {a: i for i, a in enumerate(self.domain_attrs)}
        self.marginals = {}
        for names, arr, w in marginal_list:
            m = self.Marg(names, arr, w, self.attr_index, self.domain_shape)
            self.marginals[names] = m

    def _compute_dependency(self):
        deps = {}
        for key, m in self.marginals.items():
            new = type('S', (), {})()
            new.attr_set = set(key)
            new.dependency = set()
            subsets = copy.deepcopy(deps)
            for sk, sv in subsets.items():
                inter = sorted(sv.attr_set & set(key))
                if inter:
                    it = tuple(inter)
                    if it not in deps:
                        s2 = type('S2', (), {})()
                        s2.attr_set = set(inter)
                        s2.dependency = set()
                        deps[it] = s2
                    if set(inter) != set(sk):
                        deps[sk].dependency.add(it)
                    if set(inter) != set(key):
                        new.dependency.add(it)
                    for ok, ov in deps.items():
                        if set(inter) < ov.attr_set:
                            ov.dependency.add(it)
            deps[key] = new
        return deps

    def _consist_on_subset(self, target_set, target_margs):
        names = tuple(target_set)
        size = int(np.prod([self.domain_shape[self.attr_index[a]] for a in target_set])) if target_set else 1
        common = self.Marg(names, np.zeros(size), 1.0, self.attr_index, self.domain_shape)
        common.init_consist_parameters(len(target_margs))
        for i, m in enumerate(target_margs):
            common.project_from_bigger(m, i)
        common.calculate_delta()
        for i, m in enumerate(target_margs):
            m.update_marg(common, i)

    def run(self):
        deps = self._compute_dependency()
        for m in self.marginals.values():
            m.calculate_tuple_key()
            m.get_sum()
        for _ in range(self.iterations):
            self._consist_on_subset([], list(self.marginals.values()))
            temp = copy.deepcopy(deps)
            while temp:
                key, sub = next(((k, v) for k, v in temp.items() if not v.dependency), (None, None))
                if key is None:
                    break
                target = [m for m in self.marginals.values() if sub.attr_set <= m.attr_set]
                if len(target) > 1:
                    self._consist_on_subset(sub.attr_set, target)
                    for v in temp.values():
                        if key in v.dependency:
                            v.dependency.remove(key)
                temp.pop(key, None)
            neg = False
            for m in self.marginals.values():
                if (m.count < 0).any():
                    m.non_negativity(self.non_negativity_method)
                    neg = True
            if not neg:
                break
        result = []
        for names, m in self.marginals.items():
            result.append((names, m.count, m.rho))
        params = {'consist_iterations': self.iterations, 'non_negativity': self.non_negativity_method}
        return result, params

