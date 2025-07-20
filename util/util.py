import os 
import json

def make_exp_dir(args):
    resample_fix = 'resample' if args.resample else None 
    graph_sample_fix = 'graphical' if args.graph_sample else None
    public_fix = 'pub' if args.pub else None
    test_fix = 'test' if args.test else None

    arg_list = [args.test, resample_fix, graph_sample_fix, public_fix, test_fix, args.lr, args.iter, args.batch]
    if any(param is not None or param for param in arg_list):
        postfix = '_'.join(str(param) for param in arg_list if param is not None and param)
        if len(postfix)>0:
            postfix = '_' + postfix

        parent_dir = f'exp/{args.dataset}/{args.method}/{args.epsilon}_{args.num_preprocess}_{args.rare_threshold}' + postfix
        data_path = f'data/{args.dataset}/'
        os.makedirs(parent_dir, exist_ok=True)
        print('parent_dir:', parent_dir)
    else:
        print('use default path')
        parent_dir = f'exp/{args.dataset}/{args.method}/{args.epsilon}_{args.num_preprocess}_{args.rare_threshold}'
        data_path = f'data/{args.dataset}/'
        os.makedirs(parent_dir, exist_ok=True)

    return parent_dir, data_path


def make_ablation_dir(args):
    resample_fix = 'resample' if args.resample else None 
    adap_fix = 'adaptive' if args.adaptive else None
    marg_num = str(args.marg_num) if args.marg_num > 0 else None
    iter = str(args.iter) if args.iter is not None else None

    arg_list = [resample_fix, adap_fix, marg_num, iter]

    if any(param is not None or param for param in arg_list):
        postfix = '_'.join(str(param) for param in arg_list if param is not None and param)
        if len(postfix)>0:
            postfix = '_' + postfix

        parent_dir = f'exp/{args.dataset}_ablation/{args.method}/{args.epsilon}_{args.num_preprocess}_{args.rare_threshold}' + postfix
        data_path = f'data/{args.dataset}/'
        os.makedirs(parent_dir, exist_ok=True)
        print('parent_dir:', parent_dir)
    else:
        print('use default path')
        parent_dir = f'exp/{args.dataset}_ablation/{args.method}/{args.epsilon}_{args.num_preprocess}_{args.rare_threshold}'
        data_path = f'data/{args.dataset}/'
        os.makedirs(parent_dir, exist_ok=True)
        print('parent_dir:', parent_dir)

    return parent_dir, data_path

def make_exp_dir_temp(args):
    resample_fix = 'resample' if args.resample else None 
    graph_sample_fix = 'graphical' if args.graph_sample else None
    public_fix = 'pub' if args.pub else None
    test_fix = 'test' if args.test else None

    arg_list = [args.test, resample_fix, graph_sample_fix, public_fix, test_fix, args.lr, args.iter, args.batch, args.pf]
    if any(param is not None or param for param in arg_list):
        postfix = '_'.join(str(param) for param in arg_list if param is not None and param)
        if len(postfix)>0:
            postfix = '_' + postfix

        parent_dir = f'exp/{args.dataset}/{args.method}_temp/{args.epsilon}_{args.num_preprocess}_{args.rare_threshold}' + postfix
        data_path = f'data/{args.dataset}/'
        os.makedirs(parent_dir, exist_ok=True)
        print('parent_dir:', parent_dir)
    else:
        print('use default path')
        parent_dir = f'exp/{args.dataset}/{args.method}_temp/{args.epsilon}_{args.num_preprocess}_{args.rare_threshold}'
        data_path = f'data/{args.dataset}/'
        os.makedirs(parent_dir, exist_ok=True)

    return parent_dir, data_path


def prepare_eval_config(args, parent_dir):
    with open(f'data/{args.dataset}/info.json', 'r') as file:
        data_info = json.load(file)

    config = {
        'parent_dir': parent_dir,
        'real_data_path': f'data/{args.dataset}/',
        'model_params':{'num_classes': data_info['n_classes']},
        'sample': {'seed': 0, 'sample_num': data_info['train_size']}
    }

    with open(os.path.join(parent_dir, 'eval_config.json'), 'w') as file:
        json.dump(config, file, indent=4)
    return config




def algo_method(args):
    if args.method == 'aim':
        from method.AIM.aim import aim_main 
        algo = aim_main
    elif args.method == 'merf':
        from method.DP_MERF.single_generator_priv_all import merf_main
        algo = merf_main
    elif args.method in 'ddpm':
        from method.TabDDPM.run_ddpm import ddpm_main
        algo = ddpm_main
    elif args.method == 'rapp':
        from method.RAP.main import rap_main
        algo = rap_main
    elif args.method in 'privsyn':
        from method.privsyn.run_privsyn import privsyn_main
        algo = privsyn_main
    elif args.method == 'marggan':
        from method.MargDL.main import marggan_main
        algo = marggan_main
    elif args.method == 'margdiff':
        from method.MargDL.main import margdiff_main
        algo = margdiff_main
    elif args.method == 'gem':
        from method.GEM.gem import gem_main 
        algo = gem_main 
    elif args.method == 'ctgan':
        from method.CTGAN.main import ctgan_main 
        algo = ctgan_main 
    elif args.method == 'sis':
        from method.MargDL.sis import sis_main
        algo = sis_main
    
    return algo


def algo_ablation_method(args):
    if args.method == 'aim':
        from method.AIM.aim import aim_ablation_main 
        algo = aim_ablation_main
    elif args.method in ['marggan', 'marggan_adapt']:
        from method.MargDL.ablation import marggan_ablation_main
        algo = marggan_ablation_main
    elif args.method in ['merf_fit']:
        from method.DP_MERF.single_generator_priv_all import merf_ablation_main
        algo = merf_ablation_main 
    elif args.method == 'aim_chain':
        from method.AIM.aim import aim_ablation_chain 
        algo = aim_ablation_chain
    elif args.method == 'aim_longchain':
        from method.AIM.aim import aim_ablation_longchain 
        algo = aim_ablation_longchain
    elif args.method == 'marggan_all':
        from method.MargDL.ablation import marggan_ablation_all
        algo = marggan_ablation_all
    elif args.method == 'marggan_marg':
        from method.MargDL.ablation import marggan_ablation_marg
        algo = marggan_ablation_marg 
    elif args.method == 'aim_marg':
        from method.AIM.aim import aim_ablation_marg
        algo = aim_ablation_marg
    
    return algo


def algo_temp(args):
    if args.method == 'marggan':
        from method.MargDL.ablation import marggan_temp
        return marggan_temp
    elif args.method == 'margdiff':
        from method.MargDL.ablation import margdiff_temp
        return margdiff_temp
    elif args.method == 'pgm_syn':
        from method.reconstruct_algo.combine_exp_pgm_syn import pgm_syn_main
        return pgm_syn_main
    elif args.method == 'mst':
        from method.AIM.mst import mst_main 
        return mst_main