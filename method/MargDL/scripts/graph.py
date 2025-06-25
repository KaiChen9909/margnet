import networkx as nx
import numpy as np
import torch
import math 
from functools import reduce
from operator import mul
from networkx.algorithms.chordal import complete_to_chordal_graph, chordal_graph_cliques
from itertools import product, combinations

def generate_junction_tree(nodes, marginals):
    G = nx.Graph()
    for node in nodes:
        G.add_node(node)

    for marg in marginals:
        if len(marg) == 1:
            continue
        for u, v in combinations(marg, 2):
            G.add_edge(u, v)

    H, _ = complete_to_chordal_graph(G)
    cliques = [frozenset(c) for c in chordal_graph_cliques(H)]
    
    C = nx.Graph()
    for i, c in enumerate(cliques):
        C.add_node(i, clique=c)
    for i, j in combinations(range(len(cliques)), 2):
        sep = cliques[i] & cliques[j]
        if sep:
            C.add_edge(i, j, weight=len(sep), sep=sep)
    
    T = nx.maximum_spanning_tree(C, weight='weight')
    
    jt = nx.Graph()
    for i in T.nodes:
        jt.add_node(cliques[i])
    for u, v, data in T.edges(data=True):
        c_u = cliques[u]
        c_v = cliques[v]
        jt.add_edge(c_u, c_v, sep=data['sep'])
    
    return jt


def find_candidate(
        existing_marginal: list,
        column_dim: dict,
        max_sep_dim: int
    ):
    two_way_marginals = list(combinations(column_dim.keys(), 2))
    candidates = []

    for marg in two_way_marginals:
        new_magrinals = existing_marginal + [marg]
        jt = generate_junction_tree(list(column_dim.keys()), new_magrinals)
        seps = list(nx.get_edge_attributes(jt, 'sep').values())
        
        qual = True
        for sep in seps:
            total_dim = 1
            for attr in sep:
                total_dim *= column_dim[attr]
            if total_dim > max_sep_dim:
                qual = False 
                break
        if qual:
            candidates.append(marg)
    
    return candidates


def graph_sample(logits, jt, node_names, cum_num_classes, device, num_samples=1024):
    logits_num = logits.shape[0]
    num_col = len(node_names)
    all_samples = torch.zeros((num_samples, num_col), dtype=torch.int64).to(device)

    # find subgraph
    def find_subgraphs(jt):
        subgraphs = list(nx.connected_components(jt))
        return subgraphs
    
    # deal with each subgraph iteratively
    def dfs(cliques):
        # Due to the connectivity of Junction Tree, we can start with any clique and go through all other cliques
        root = list(cliques)[0]
        visited_clique = set()
            
        # sample for root clique
        root_indices = torch.randint(0, logits_num, (num_samples,))
        visited_clique.add(root)
        for node in root:
            idx = np.where(node_names == node)[0][0]
            # print('generate node:', node, ', index is', idx)

            start, end = cum_num_classes[idx], cum_num_classes[idx + 1]
            logits_node = logits[root_indices, start:end]
            all_samples[:, idx] = torch.multinomial(logits_node, 1, replacement=True).squeeze()
            

        # sample for children cliques
        def recurse(c):
            for ch in jt.neighbors(c):
                if ch in visited_clique:
                    continue 
                visited_clique.add(ch)

                sep_list = list(jt.edges[c, ch]['sep'])
                sample_logits_list = []

                for sep in sep_list:
                    target_idx = np.where(node_names == sep)[0][0]
                    target_start, target_end = cum_num_classes[target_idx], cum_num_classes[target_idx + 1]
                    target_sample = all_samples[:, target_idx] 
                    target_logits_infer = logits[:, target_start: target_end].T
                    sample_logits_list.append(target_logits_infer[target_sample, :].log())

                target_logits_all = torch.sum(torch.stack(sample_logits_list, dim=0), dim=0)
                target_logits_all = torch.clamp(target_logits_all, min=-30, max=0)
                sample_idx = torch.multinomial(target_logits_all.exp(), num_samples=1, replacement=True).squeeze()
                
                # conditional sample
                for node in ch:
                    if node in sep_list:
                        continue
                    
                    idx = np.where(node_names == node)[0][0]
                    # print('generate node:', node, ', index is', idx)
                    start, end = cum_num_classes[idx], cum_num_classes[idx + 1]
                    logits_node = logits[sample_idx, start:end]
                    all_samples[:, idx] = torch.multinomial(logits_node, num_samples=1, replacement=True).squeeze()

                recurse(ch)

        recurse(root)
    

    subgraphs = find_subgraphs(jt)
    print(subgraphs)
    for subgraph in subgraphs:
        dfs(subgraph)

    return all_samples.cpu().detach().numpy()



def find_clique_connection(jt: nx.Graph, a, b):
    '''
    For any two nodes, we will find how they correlated in junction tree by some seperators
    '''
    cliques = list(jt.nodes)
    cliques_a = [C for C in cliques if a in C]
    cliques_b = [C for C in cliques if b in C]

    for C in cliques_a:
        if C in cliques_b:
            return [(a, b)], 'simple_connected'

    best_path = None
    best_len = float('inf')
    for CA in cliques_a:
        for CB in cliques_b:
            try:
                path = nx.shortest_path(jt, CA, CB)
                if len(path) < best_len:
                    best_len = len(path)
                    best_path = path
            except nx.NetworkXNoPath:
                pass

    if best_path:
        seps = []
        for i in range(len(best_path) - 1):
            edge_data = jt.edges[best_path[i], best_path[i+1]]
            sep_set = edge_data['sep']
            seps.append(list(sep_set))

        final_list = []
        n = len(best_path)
        for i in range(n):
            if i == 0:
                clique_nodes = [a] + [seps[0]]
            elif i == n - 1:
                clique_nodes = [seps[i-1]] + [b]
            else:
                clique_nodes = [seps[i-1]] + [seps[i]]
            final_list.append(clique_nodes)

        return final_list, 'long_connected'

    return [[a], [b]], 'independent'


def chain_joint_from_cliques(connect_list, clique_logits):
    """
    Chain a minimal connect_list of joint‐distribution tensors to get
    the joint over the first left‐vars and the last right‐vars.

    Parameters
    ----------
    connect_list : list of [left, right]
        connect_list[0] = [root_vars, sep_vars]
        connect_list[i>0] = [sep_vars, new_vars]
        where root_vars/sep_vars/new_vars may be str or list of str.
    clique_logits : list of Tensors
        clique_logits[i].shape == (*dims(left_i), *dims(right_i)).

    Returns
    -------
    Tensor of shape (*dims(left_0), *dims(right_last))
        The unnormalized joint distribution over the very first left
        and the very last right variables.
    """
    # start from the first clique: p(left0, sep0)
    cur = clique_logits[0]

    # fold in each subsequent clique
    for i in range(1, len(connect_list)):
        sep, new = connect_list[i]
        nxt = clique_logits[i]

        # determine how many axes are the "new" variables
        r_ndim = len(new) if isinstance(new, list) else 1

        # normalize to get p(new | sep) along the last r_ndim axes
        # sum over new axes
        sum_axes = tuple(range(nxt.ndim - r_ndim, nxt.ndim))
        denom = nxt.sum(dim=sum_axes, keepdim=True)
        cond = nxt / denom  # now cond is p(new | sep)

        # contract out the separator dims:
        #   cur’s last s_ndim axes  ⟷  cond’s first s_ndim axes
        s_ndim = cond.ndim - r_ndim
        axes_cur  = list(range(cur.ndim - s_ndim, cur.ndim))
        axes_cond = list(range(s_ndim))
        cur = torch.tensordot(cur, cond, dims=(axes_cur, axes_cond))
        # after this, cur.shape == (*dims(left_0), *dims(new_i))

    # normalize final joint to sum to 1
    return cur / cur.sum()


def graph_marginal_query(marginal, logits, jt, node_names, cum_num_classes, device):
    assert (
        len(marginal) <= 2
    ), "Only support 2 and lower way marginals" 

    if len(marginal) == 1:
        node = marginal[0]
        start_idx = cum_num_classes[np.where(node_names == node)[0][0]]
        end_idx = cum_num_classes[np.where(node_names == node)[0][0] + 1]

        joint_prob = logits[:, start_idx: end_idx].mean(dim=0).to(device)
        return joint_prob.detach().cpu().numpy().flatten() 

    else:
        connect_list, connect_status = find_clique_connection(jt, marginal[0], marginal[1])

        if connect_status == 'simple_connected':
            start_idx = [cum_num_classes[np.where(node_names == node)[0][0]] for node in marginal]
            end_idx = [cum_num_classes[np.where(node_names == node)[0][0] + 1] for node in marginal]

            z_splits = [logits[:, start:end] for start, end in zip(start_idx, end_idx)]

            input_dims = ','.join(f'b{chr(105 + i)}' for i in range(len(z_splits)))
            output_dims = ''.join(f'{chr(105 + i)}' for i in range(len(z_splits)))
            einsum_str = f'{input_dims}->b{output_dims}'
            joint_prob = torch.einsum(einsum_str, *z_splits).mean(dim=0).to(device)

            return joint_prob.detach().cpu().numpy().flatten() 
        
        elif connect_status == 'independent': 
            start_idx = [cum_num_classes[np.where(node_names == node)[0][0]] for node in marginal]
            end_idx = [cum_num_classes[np.where(node_names == node)[0][0] + 1] for node in marginal]

            z_splits = [logits[:, start:end].exp().mean(dim=0) for start, end in zip(start_idx, end_idx)]
            joint_prob = torch.einsum('i,j->ij', *z_splits).to(device)

            return joint_prob.detach().cpu().numpy().flatten() 

        elif connect_status == 'long_connected':
            dims = {}
            for i, node in enumerate(node_names):
                dims[node] = cum_num_classes[i+1] - cum_num_classes[i]
            
            clique_logits = []
            for cl in connect_list:
                flatten_cl = [x for elt in cl for x in (elt if isinstance(elt, list) else [elt])]

                start_idx = [cum_num_classes[np.where(node_names == node)[0][0]] for node in flatten_cl]
                end_idx = [cum_num_classes[np.where(node_names == node)[0][0] + 1] for node in flatten_cl]

                z_splits = [logits[:, start:end] for start, end in zip(start_idx, end_idx)]

                input_dims = ','.join(f'b{chr(105 + i)}' for i in range(len(z_splits)))
                output_dims = ''.join(f'{chr(105 + i)}' for i in range(len(z_splits)))
                einsum_str = f'{input_dims}->b{output_dims}'
                joint_logits = torch.einsum(einsum_str, *z_splits).mean(dim=0).to(device)

                clique_logits.append(joint_logits)
            
            joint_prob = chain_joint_from_cliques(connect_list, clique_logits)
            joint_prob = joint_prob/joint_prob.sum()

            return joint_prob.detach().cpu().numpy().flatten() 