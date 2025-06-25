import sys
target_path="./"
sys.path.append(target_path)

import os
import argparse
from copy import deepcopy
from method.TabDDPM.scripts.pretrain_and_finetune import pretrain, finetune
from method.TabDDPM.scripts.sample import ddpm_sampler
from method.TabDDPM.scripts.neighbor import *
from method.TabDDPM.data.dataset import * 
from method.TabDDPM.data.data_utils import * 


def ddpm_main(args, df, domain, rho, parent_dir, **kwargs):
    # basic config
    if args.epsilon > 0 and not args.test:
        epsilon = args.epsilon
        delta = args.delta 
    else:
        epsilon = None 
        delta = None
    
    print(f'total privacy budget: ({epsilon}, {delta})')

    base_config_path = f'method/TabDDPM/exp/{args.dataset}/config.toml'
    base_config = load_config(base_config_path)

    data_info = load_json(f'data/{args.dataset}/info.json')
    dataset = make_dataset_from_df(
            df,
            T = Transformations(**base_config['train']['T']),
            y_num_classes = base_config['model_params']['num_classes'],
            is_y_cond = base_config['model_params']['is_y_cond'],
            task_type = data_info['task_type'],
            df_pub = kwargs.get('df_pub', None)
        )
    
    rho_used = kwargs.get('rho_used', None)
    train_size = len(dataset.y['train'])
    base_config["parent_dir"] = parent_dir
    base_config['sample']['num_samples'] = train_size
    dump_config(base_config, f'{parent_dir}/config.toml')

    # fit the diffusion model
    diffusion_model = finetune(
            **base_config['train']['main'],
            **base_config['diffusion_params'],
            parent_dir=base_config['parent_dir'],
            dataset = dataset,
            model_type=base_config['model_type'],
            model_params=base_config['model_params'],
            model_path = None,
            T_dict=base_config['train']['T'],
            num_numerical_features=base_config['num_numerical_features'],
            device=args.device,
            dp_epsilon = epsilon,
            dp_delta = delta,
            rho_used = rho_used,
            report_every = args.test
        ) 
    
    sampler = ddpm_sampler(
        diffusion=diffusion_model,
        num_numerical_features=base_config['num_numerical_features'],
        T_dict = base_config['train']['T'],
        dataset = dataset,
        model_params = base_config['model_params']
    )

    return {"ddpm_generator": sampler} 


def pre_ddpm_main(args, df, domain, rho, parent_dir, **kwargs):
    # basic config
    if args.epsilon > 0:
        epsilon = args.epsilon
        delta = args.delta 
    else:
        epsilon = None 
        delta = None
    
    print(f'total privacy budget: ({epsilon},{delta})')

    base_config_path = f'method/TabDDPM/exp/{args.dataset}/config.toml'
    base_config = load_config(base_config_path)

    data_info = load_json(f'data/{args.dataset}/info.json')
    dataset = make_dataset_from_df(
            df,
            T = Transformations(**base_config['train']['T']),
            y_num_classes = base_config['model_params']['num_classes'],
            is_y_cond = base_config['model_params']['is_y_cond'],
            task_type = data_info['task_type'],
            df_pub = kwargs.get('df_pub', None)
        )
    
    rho_used = kwargs.get('rho_used', None)
    train_size = len(dataset.y['train'])
    base_config["parent_dir"] = parent_dir
    base_config['sample']['num_samples'] = train_size
    dump_config(base_config, f'{parent_dir}/config.toml')

    # fit the diffusion model
    if 'pretrain' in dataset.y.keys():
        neighbor_cosine_sample(dataset, 0.1 * rho, train_size)
        if not rho_used:
            rho_used += 0.1 * rho
        else:
            rho_used = 0.1 * rho
        diffusion_model = pretrain(
                **base_config['pretrain']['main'],
                **base_config['diffusion_params'],
                parent_dir=base_config['parent_dir'],
                dataset = dataset,
                model_type=base_config['model_type'],
                model_params=base_config['model_params'],
                T_dict=base_config['train']['T'],
                num_numerical_features=base_config['num_numerical_features'],
                device=args.device,
                report_every = args.test
        )

    diffusion_model = finetune(
            **base_config['train']['main'],
            **base_config['diffusion_params'],
            parent_dir=base_config['parent_dir'],
            dataset = dataset,
            model_type=base_config['model_type'],
            model_params=base_config['model_params'],
            model_path = os.path.join(parent_dir, 'pretrain_model.pt') if 'pretrain' in dataset.y.keys() else None,
            T_dict=base_config['train']['T'],
            num_numerical_features=base_config['num_numerical_features'],
            device=args.device,
            dp_epsilon = epsilon,
            dp_delta = delta,
            rho_used = rho_used,
            report_every = args.test
        ) 



    sampler = ddpm_sampler(
        diffusion=diffusion_model,
        num_numerical_features=base_config['num_numerical_features'],
        T_dict = base_config['train']['T'],
        dataset = dataset,
        model_params = base_config['model_params']
    )

    return {"ddpm_generator": sampler} 



def pe_ddpm_main(args, df, domain, rho, parent_dir, **kwargs):
    # basic config
    if args.epsilon > 0:
        epsilon = args.epsilon
        delta = args.delta 
    else:
        epsilon = None 
        delta = None
    
    print(f'total privacy budget: ({epsilon},{delta})')

    base_config_path = f'method/TabDDPM/exp/{args.dataset}/config.toml'
    base_config = load_config(base_config_path)

    data_info = load_json(f'data/{args.dataset}/info.json')
    dataset = make_dataset_from_df(
            df,
            T = Transformations(**base_config['train']['T']),
            y_num_classes = base_config['model_params']['num_classes'],
            is_y_cond = base_config['model_params']['is_y_cond'],
            task_type = data_info['task_type'],
            df_pub = kwargs.get('df_pub', None)
        )
    
    rho_used = kwargs.get('rho_used', None)
    train_size = len(dataset.y['train'])
    base_config["parent_dir"] = parent_dir
    base_config['sample']['num_samples'] = train_size
    dump_config(base_config, f'{parent_dir}/config.toml')

    # fit the diffusion model
    if 'pretrain' in dataset.y.keys():
        model_path = None
        pretrain_size = dataset.y['pretrain'].shape[0]
        for i in range(5):
            diffusion_model = pretrain(
                    **base_config['pretrain']['main'],
                    **base_config['diffusion_params'],
                    parent_dir=base_config['parent_dir'],
                    dataset = dataset,
                    model_type=base_config['model_type'],
                    model_params=base_config['model_params'],
                    model_path = model_path,
                    T_dict=base_config['train']['T'],
                    num_numerical_features=base_config['num_numerical_features'],
                    device=args.device,
                    report_every = args.test
            )
            sampler = ddpm_sampler(
                diffusion=diffusion_model,
                num_numerical_features=base_config['num_numerical_features'],
                T_dict = base_config['train']['T'],
                dataset = dataset,
                model_params = base_config['model_params']
            )
            dataset.update_pretrain_data(
                df_new = sampler.sample(num_sample=2 * dataset.y['train'].shape[0], device=args.sample_device)
            )
            neighbor_freq_sample(dataset, 0.1 * rho, pretrain_size)
            if not rho_used:
                rho_used += 0.1 * rho
            else:
                rho_used = 0.1 * rho
            model_path = os.path.join(parent_dir, 'pretrain_model.pt')


    diffusion_model = finetune(
            **base_config['train']['main'],
            **base_config['diffusion_params'],
            parent_dir=base_config['parent_dir'],
            dataset = dataset,
            model_type=base_config['model_type'],
            model_params=base_config['model_params'],
            model_path = os.path.join(parent_dir, 'pretrain_model.pt') if 'pretrain' in dataset.y.keys() else None,
            T_dict=base_config['train']['T'],
            num_numerical_features=base_config['num_numerical_features'],
            device=args.device,
            dp_epsilon = epsilon,
            dp_delta = delta,
            rho_used = rho_used,
            report_every = args.test
        ) 



    sampler = ddpm_sampler(
        diffusion=diffusion_model,
        num_numerical_features=base_config['num_numerical_features'],
        T_dict = base_config['train']['T'],
        dataset = dataset,
        model_params = base_config['model_params']
    )

    return {"ddpm_generator": sampler}
    
