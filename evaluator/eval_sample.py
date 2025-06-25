import os


def eval_sampler(
        sampling_method, 
        temp_config, 
        device=None, 
        preprocesser=None,
        **kwargs
    ):
    if sampling_method == 'merf':
        kwargs.get("merf_generator").sample(
            n = temp_config['sample']['sample_num'],
            preprocessor = preprocesser,
            parent_dir = temp_config['parent_dir'],
            device = device
        )
    # sample via privsyn
    elif sampling_method == 'privsyn':
        kwargs.get("privsyn_generator").syn(
            n_sample = temp_config['sample']['sample_num'],
            preprocesser = preprocesser,
            parent_dir = temp_config['parent_dir']
        )
    
    # sample via aim 
    elif sampling_method == 'aim':
        kwargs.get("aim_generator").syn_data(
            num_synth_rows = temp_config['sample']['sample_num'],
            path = temp_config['parent_dir'],
            preprocesser = preprocesser
        )
    
    elif sampling_method == 'mrf':
        kwargs.get("mrf_generator").syn(
            n_sample = temp_config['sample']['sample_num'],
            preprocesser = preprocesser,
            path = temp_config['parent_dir']
        )

    elif sampling_method in ['rap', 'rapp']:
        kwargs.get("RAP_generator").syn(
            n_sample = temp_config['sample']['sample_num'],
            preprocesser = preprocesser,
            parent_dir = temp_config['parent_dir']
        )

    elif sampling_method == 'gem':
        kwargs.get("gem_generator").syn(
            n_sample = temp_config['sample']['sample_num'],
            # n_sample = 100,
            preprocesser = preprocesser,
            parent_dir = temp_config['parent_dir']
        )

    elif sampling_method in ['ddpm']:
        kwargs.get("ddpm_generator").sample(
            num_sample = temp_config['sample']['sample_num'],
            preprocesser = preprocesser,
            parent_dir = temp_config['parent_dir'],
            device = device,
            seed = temp_config['sample']['seed']
        ) 
    
    elif sampling_method in ['sis', 'marggan', 'margdiff']:
        kwargs.get("MargDL_generator").sample(
            num_samples = temp_config['sample']['sample_num'],
            preprocesser = preprocesser,
            parent_dir = temp_config['parent_dir']
        ) 

    elif sampling_method == 'ctgan':
        kwargs.get("ctgan_generator").sample(
            num_samples = temp_config['sample']['sample_num'],
            preprocesser = preprocesser,
            parent_dir = temp_config['parent_dir']
        ) 