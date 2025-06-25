from method.CTGAN.scripts.dp_ctgan_rdp import DPCTGANSynthesizerRDP


def construct_ctgan_args(args, domain):
    embedding_dim = 0
    for k,v in domain.items():
        if k.split('_')[0] == 'num':
            embedding_dim += 1
        else:
            embedding_dim += v
    args.embedding_dim = embedding_dim

    if args.dataset in ['adult', 'gauss10']:
        args.model_dim = (256, 256)
    elif args.dataset == 'bank':
        args.model_dim = (512, 512)
    elif args.dataset in ['loan', 'gauss30']:
        args.model_dim = (1024, 1024)
    else:
        args.model_dim = (512, 512)

    return args



def ctgan_main(args, df, domain, rho, **kwargs):
    rho_used = kwargs.get('rho_used', 0)
    args = construct_ctgan_args(args, domain)

    cat_col = []
    for key in domain.keys():
        if key.split('_')[0] in ['cat', 'y']:
            cat_col.append(key)
    cat_col = tuple(cat_col)

    model = DPCTGANSynthesizerRDP(
        embedding_dim=args.embedding_dim,
        generator_dim=args.model_dim,
        discriminator_dim=args.model_dim,
        device=args.device,
        target_epsilon=args.epsilon, 
        target_delta=args.delta, 
        rho_total = rho + rho_used, 
        rho_used = rho_used, 
        domain = domain
    )
    
    model.fit(
        train_data = df,
        discrete_columns = cat_col
    )

    return {'ctgan_generator': model}

