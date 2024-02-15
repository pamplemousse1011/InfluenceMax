# SOMEHOW V2 LIMIT THE EXPLORATION: 
# IT TRAINS NN AND ACQUIRES THE XMIN(NN)
# IT THEN RETRAINS THE NN WITH XMIN(NN), FXMIN(NN) INCLUDED
# ALGORIGHM ACQUIRS XNEW AFTERWARDS
# XNEW VERY CLOSE TO XMIN(NN)
# V1 ACQUIRES XMIN(NN) AFTER XNEW FROM ALGORIGHM, 
# IT ONLY RETRAIN NETWORKS ONCE 
# DOES THAT MEAN THE ALGORITHM IS MOSTLY EXPLOITATION?

import argparse 
import copy
import os

import numpy as np
import jax.numpy as jnp 
import jax.random as jr
jr.PRNGKey(42)

import torch
import functools
import blackhc.laaos as laaos 
from blackhc.laaos import create_file_store

from codes.utils import print_x, print_progress, gc_cuda, ignore_criterion, perturb
from codes.context_stopwatch import ContextStopwatch
import codes.data_modules.data_generator as OptGenData 
from codes.influence_max.hyperparam_optimization.influence_max import InfluenceMax  
from codes.influence_max.hyperparam_optimization.opt_train_pl import train_pl_model

def create_experiment_config_argparser(parser):
    parser.add_argument(
        "--use_double", 
        action="store_true", 
        default=False, 
        help="whether to use double precision; if not, float"
    )

    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=51200, 
        help="input batch size for training"
    ) 

    parser.add_argument(
        "--max_epochs", 
        type=int, 
        default=1000, 
        help="maximum number of epochs to train"
    )

    parser.add_argument(
        "--min_epochs", 
        type=int, 
        default=1, 
        help="minimum number of epochs to train"
    )
    
    parser.add_argument(
        "--sampler",
        type=str,
        default="none",
        help="choice of 'none' or 'random_fixed_length'"
    )

    parser.add_argument(
        "--epoch_samples", 
        type=int, 
        default=1024,
        help="number of epochs to train"
    )
    
    parser.add_argument(
        "--available_sample_k",
        type=int,
        default=1,
        help="number of active samples to add per active learning iteration",
    )

    parser.add_argument(
        "--n_initial_samples",
        type=int,
        default=20,
        help="number of initial active samples before the active optimization starts",
    )

    parser.add_argument(
        "--target_num_acquired_samples", 
        type=int, 
        default=100, 
        help="max number of samples to acquire"
    )

    parser.add_argument(
        "--target_accuracy", 
        type=float, 
        default=0.80, 
        help="max accuracy to train to"
    )

    parser.add_argument(
        "--target_tolerance", 
        type=float, 
        default=1e-4, 
        help="min tolerance to train to"
    )

    parser.add_argument(
        "--acquisition_method", 
        type=str, 
        default="infmax", 
        help="acquisition method can be 'infmax' or 'random'."
    )

    parser.add_argument(
        "--no_cuda", 
        action="store_true", 
        default=False, 
        help="disables CUDA training"
    )

    parser.add_argument(
        "--n_devices", 
        type=int, 
        default=1, 
        help="Number of devices used in trainer"
    )

    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=1, 
        help="Number of workers used in dataloader"
    )

    parser.add_argument(
        "--seed", 
        type=int, 
        default=6, 
        help="random seed"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="CIFAR10",
        help=f"CIFAR10",
    )

    parser.add_argument(
        "--path_laaos_prefix",
        type=str,
        default='./data-out/', 
        help="path of the prefix for laaos",
    )

    parser.add_argument(
        "--path_logs",
        type=str,
        default='./logs/',
        help="path of the lightning logs",
    )
    
    parser.add_argument(
        "--path_data",
        type=str,
        default='./data/',
        help="path of the downloaded datasets, such as MNIST, CIFAR10, and etc.",
    )

    parser.add_argument(
        "--n_candidate_model",
        type=int,
        default=3,
        help="number of candidate models to estimate y",
    )

    parser.add_argument(
        "--n_ensemble_model",
        type=int,
        default=3,
        help="number of iterations/models to compute (jacknife/ensemble) estimator of Ey",
    ) 

    parser.add_argument(
        "--use_validation_set",
        action="store_true",
        default=False,
        help="whether use validation set",
    )

    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3, 
        help="whether perform early_stopping",
    )

    parser.add_argument(
        "--check_val_every_n_epoch",
        type=int,
        default=50, 
        help="One check happens after every _ numbers of training epochs",
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
        help="learning rate used in optim.AdamW for training",
    )
 
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="weight_decay in the optim.AdamW for training",
    )
    
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="gamma used in lr_scheduler.MultiStepLR for training",
    )  

    parser.add_argument(
        "--n_hidden",
        type=int,
        nargs='+',
        default=[512, 64, 64, 64],
        help="Number of hidden units in each layer of the neural networks",
    )

    parser.add_argument(
        "--sto_n_resample",
        type=int,
        default=50,
        help="number of resamples to average over for stochastic MLP",
    )

    parser.add_argument(
        "--sto_n_noise",
        type=int,
        default=50,
        help="number of noise neurons in the stochastic layer",
    )

    parser.add_argument(
        "--sto_noise_std",
        type=float,
        default=1.,
        help="std of noise neurons in the stochastic layer",
    )

    parser.add_argument(
        "--no_batch_norm",
        action="store_true", 
        default=False, 
        help="disable batch norm if True"
    )

    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0,
        help="Probability of an element to be zeroed",
    ) 
    
    parser.add_argument(
        "--ihvp_method",
        type=str,
        default='cg-linalg',
        help="Method to compute the inverse hessian vector product, conjugate gradient 'cg' or LiSSA 'lissa'",
    )

    parser.add_argument(
        "--ihvp_batch_size", 
        type=int, 
        default=5000, 
        help="input batch size for hessian in inverse hessian product estimation"
    )
    
    parser.add_argument(
        "--cg_method",
        type=str,
        default='trust-ncg',
        help="optimization method for conjugate gradient; choice: 'Newton-CG', 'trust-ncg', 'trust-krylov'.",
    )

    parser.add_argument(
        "--cg_lambda",
        type=float,
        default=0.001,
        help="The damping term to stabilize the solution for conjugate gradient",
    )

    parser.add_argument(
        "--lissa_T",
        type=int,
        default=1,
        help="number of independent runs whose results to be averaged to compute inverse hvp",
    )

    parser.add_argument(
        "--lissa_J",
        type=int,
        default=50,
        help="number of recursive computation so that approximation converges in inverse hvp computation",
    )

    parser.add_argument(
        "--lissa_scaling",
        type=float,
        default=1e-4,
        help="scaling factor in lissa inverse hessian product estimation",
    )

    parser.add_argument(
        "--lissa_damping",
        type=float,
        default=0,
        help="damping factor in lissa inverse hessian product estimation",
    )
    
    parser.add_argument(
        "--scaling_task",
        type=float,
        default=1e-06,
        help="scaling factor in hessian computation for task gradients",
    )
    
    parser.add_argument(
        "--search_xmin_nstart",
        type=int,
        default=100,
        help="jow many times we initialize the starting points for optimization in xmin search",
    )

    parser.add_argument(
        "--search_xmin_opt_tol",
        type=float,
        default=1e-03,
        help="it determines the optimization algorithm will stop when either the gradient norm or the step size falls below tol, depending on which of these conditions is relevant for the chosen method",
    )
    
    parser.add_argument(
        "--search_xmin_method",
        type=str,
        default='grid-search',
        help="method in xmin search; choice includes 'multi-start' and 'grid-search'",
    )

    parser.add_argument(
        "--search_xmin_opt_method",
        type=str,
        default='trust-constr',
        help="optimization method in xmin search; choice includes 'L-BFGS-B', 'TNC', 'SLSQP', 'Powell' and 'trust-constr'",
    )

    parser.add_argument(
        "--disp",
        action="store_true", 
        default=True, 
        help="set to True to print convergence messages",
    )
    
    parser.add_argument(
        "--trans_method",
        type=str,
        default='rbf',
        help="transform method",
    )
    
    parser.add_argument(
        "--trans_rbf_nrad",
        type=int,
        default=5,
        help="number of dimensions to expand in radial basis functions",
    ) 
    
    parser.add_argument(
        "--do_normalize_output",
        action="store_true", 
        default=False, 
        help="whether to normalize the output",
    )
   
    parser.add_argument(
        "--n_small_data",
        type=int,
        default=10000, 
        help="number of base data to make a small training base dataset",
    )

    parser.add_argument(
        "--n_select_base",
        type=int,
        default=10000, 
        help="number of base data to train the infmax model",
    ) 
    
    parser.add_argument(
        "--progress_bar",
        action="store_true", 
        default=False, 
        help="whether print the progress",
    )

    parser.add_argument(
        "--trial",
        type=int,
        default=1,
        help="_-th trial",
    ) 
    
    parser.add_argument(
        "--resume",
        action="store_true", 
        default=False, 
        help="whether to resume from the previous results",
    ) 

    parser.add_argument(
        "--net_type",
        type=str,
        default="preactrn",
        help=f"vgg,lenet,cnn,resnet,preactrn,wrn",
    )

    parser.add_argument(
        "--covariate_shift",
        action="store_true", 
        default=False, 
        help="set to True to activate covariate shift",
    )

    parser.add_argument(
        "--shift_in_distribution",
        action="store_true", 
        default=False, 
        help="set to True to generate in distribution shifted test data",
    )

    parser.add_argument(
        "--ignore_threshold",
        type=float,
        default=0.05,
        help="if any of the acquired samples has all dimensions with \
              its absolute difference of the candidate's corresponding \
              dimension smaller than the threshold, then we should ignore. \
              The default is we should not ignore at all.",
    ) 

    parser.add_argument(
        "--use_pretrained_featemb",
        action="store_true", 
        default=False, 
        help="use the last step feature embedding model or train to obtain the feature embedding (paired with disable_base_x_embedding_training)...."
    )

    parser.add_argument(
        "--pretrained_featemb_model_path",
        type=str,
        default="./pretrained_model/resnet18.py", 
        help="full path to get pretrained_featemb file, e.g. '/usr/pretrained_model/resnet18.pt'"
    )

    parser.add_argument(
        "--pretrained_featemb_use_logits",
        action="store_true", 
        default=False, 
        help="use the logits from the pretrained model or use the feature embedding from the pretrained model as the embedding."
    )

    parser.add_argument(
        "--disable_base_x_embedding_training",
        action="store_false", 
        default=True, 
        help="disable the base_x_embedding training. If True then use_pretrained_featemb has to be True. If False and paired with use_pretrained_featemb, then train a linear layer on top of feature embedding from the pretrained model, otherwise train from scratch."
    ) 

    parser.add_argument(
        "--use_stochastic",
        action="store_true", 
        default=False, 
        help="whether to use stochastic layer for infmax model"
    )

    return parser
 
def main():
    parser = argparse.ArgumentParser(
        description="InfMax", formatter_class=functools.partial(argparse.ArgumentDefaultsHelpFormatter, width=120)
    )
    
    parser.add_argument(
        "--experiment_description", 
        type=str, 
        default="Active optimization", 
        help="Description of the experiment"
    )
    parser = create_experiment_config_argparser(parser)
    args, _ = parser.parse_known_args()
     
    ## Setting compute 
    args.use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.pin_memory = True if args.use_cuda else False

    store_name = '_'.join([
        '-'.join([args.dataset, args.net_type]),
        'h0l0_var0.0', 
        'bsize' + str(args.available_sample_k),  
        'seed' + str(args.seed),
        'trial'+ str(args.trial),
        args.acquisition_method
    ])

    resume_filename = os.path.join(args.path_laaos_prefix, store_name+'.py')
    
    # Initial samples are assumed yet-to-be-generated
    initial_to_generate = True
    # Acquisition step is initialized to be 0
    initial_iterations = []
    it = 0
    
    if args.resume and os.path.isfile(resume_filename):
        print(f"Loading previous file {resume_filename}... ")

        resume_file = laaos.safe_load(resume_filename)
        resume_args = resume_file['args']

        path_laaos_prefix = args.path_laaos_prefix
        target_num_acquired_samples = args.target_num_acquired_samples
        use_cuda    = args.use_cuda
        pin_memory  = args.pin_memory
        num_workers = args.num_workers
        n_devices   = args.n_devices
        
        args = resume_args
        args['target_num_acquired_samples'] = target_num_acquired_samples
        args['path_laaos_prefix'] = path_laaos_prefix
        args['num_workers'] = num_workers
        args['pin_memory']  = pin_memory
        args['use_cuda']    = use_cuda
        args['n_devices']   = n_devices
        dtype = np.float64 if args['use_double'] else np.float32

        if 'initial_samples' in resume_file and len(resume_file['initial_samples']) > 0:
            initial_to_generate = False 
            initial_samples = resume_file['initial_samples']
            samples_x       = np.array(initial_samples['init_x'],       dtype=dtype)
            samples_y_train = np.array(initial_samples['init_y_train'], dtype=dtype)
            samples_y_targt = np.array(initial_samples['init_y_targt'], dtype=dtype) 
            
            gendata = getattr(OptGenData, args['dataset'])(
                use_double=args['use_double'],
                net_type=args['net_type'],
                n_small_data=args['n_small_data'],
                data_dir=args['path_data'],
                log_dir=args['path_logs'],
                seed=args['seed'],
                shift=args.get('covariate_shift', False),
                in_distribution=args.get('shift_in_distribution', False),
                num_workers=args['num_workers'],
                pin_memory=args['pin_memory'],
                n_devices=args['n_devices']
            )
            if 'iterations' in resume_file and len(resume_file['iterations']) > 0:
                initial_iterations = resume_file['iterations']
                # iteration recording at step 0, so needs to exclude the first one
                for item in initial_iterations[1:]:
                    chosen_samples  = np.array(item['chosen_samples'], dtype=dtype)
                    samples_x       = np.vstack([samples_x, chosen_samples]) 
                    samples_y_train = np.hstack([samples_y_train, item['y_train_chosen_samples']]).astype(dtype)
                    samples_y_targt = np.hstack([samples_y_targt, item['y_targt_chosen_samples']]).astype(dtype)
                                        
                lastitem = initial_iterations[-1]
                it = lastitem['step'] 

                if lastitem['step'] >= target_num_acquired_samples:
                    print("DONE.")
                    return        
        
    else:
        print(f"Creating new file {store_name}... ")
        args = args.__dict__
        dtype = np.float64 if args['use_double'] else np.float32
    print(args)

    ## Setting compute  
    device = "cuda" if args['use_cuda'] else "cpu"
    print(f"Using {device} for computations.")

    if initial_to_generate:
        gendata = getattr(OptGenData, args['dataset'])(
            seed=args['seed'], 
            use_double=args['use_double'],
            n_small_data=args['n_small_data'],
            net_type=args['net_type'],
            data_dir=args['path_data'],
            log_dir=args['path_logs'],
            shift=args.get('covariate_shift', False),
            in_distribution=args.get('shift_in_distribution', False),
            num_workers=args['num_workers'],
            pin_memory=args['pin_memory']
        )
        
        # generate initial samples
        samples_x = gendata.gen_samples(args['n_initial_samples'])
        samples_y_train = []
        samples_y_targt = [] 
        for ii in range(args['n_initial_samples']):
            # call the base model 
            out_train, out_targt = gendata.evaluate_all_and_save_individual(samples_x[ii], ii)
            samples_y_train.append(out_train)  
            samples_y_targt.append(out_targt) 
        
        samples_y_train = np.hstack(samples_y_train).astype(samples_x.dtype)
        samples_y_targt = np.hstack(samples_y_targt).astype(samples_x.dtype) 
            
        initial_samples = {'init_x'  : samples_x.tolist(), 
                           'init_y_train': samples_y_train.tolist(), 
                           'init_y_targt': samples_y_targt.tolist()}  

    ## Current status
    opt_target = gendata._f_opt
    # the i-th sample that gives the best loss value on dataset A 
    idx_obsmin = np.argmin(samples_y_train)
    # the corresponding hyperparameters
    x_obsmin   = samples_x[idx_obsmin]
    # the best loss on dataset A   
    tol_train_obsmin = samples_y_train[idx_obsmin] - opt_target
    # the corresponding loss on dataset B
    tol_targt_obsmin  = samples_y_targt[idx_obsmin] - opt_target

    tol = copy.deepcopy(tol_targt_obsmin)
           
    print_progress(
        nsample   = samples_x.shape[0],
        tol       = tol_targt_obsmin,  
        train_tol = tol_train_obsmin, 
        xmin      = x_obsmin
    )

    store = create_file_store(
        store_name,
        suffix="",
        prefix=args['path_laaos_prefix'],
        truncate=False,
        type_handlers=(laaos.StrEnumHandler(), laaos.ToReprHandler()),
    )
    store["args"] = args
    store["initial_samples"] = initial_samples
    # store wraps the empty list in a storable list, so we need to fetch it separately.
    store["iterations"] = []
    ## The best results from the initial samples 
    iterations = store["iterations"]

    if it == 0:
        iterations.append(
            dict(
                step=0, 
                tol=tol_targt_obsmin,  
                train_tol=tol_train_obsmin, 
                xmin=x_obsmin.tolist(), 
                scores=['random'],
                chosen_samples         = samples_x[-args['available_sample_k']:].reshape(args['available_sample_k'], -1).tolist(), 
                y_train_chosen_samples = samples_y_train[-args['available_sample_k']:].tolist(), 
                y_targt_chosen_samples = samples_y_targt[-args['available_sample_k']:].tolist(), 
                train_model_elapsed_time=0,
                batch_acquisition_elapsed_time=0,
            )
        ) 
    else:
        ## The best results from the following acquisitions, if there is any
        for item in initial_iterations:
            iterations.append(item)
    
    # Updating the number of initial sample points
    n_init = samples_x.shape[0]
    print(f"Starting from {n_init} number of samples, intend to acquire another {args['target_num_acquired_samples']}.")
    
    ignore_flag, ignore_flag_fxmin = True, True
    while (samples_x.shape[0] - args['n_initial_samples'] < args['target_num_acquired_samples']): 
        if tol <= args['target_tolerance']:
            print("Reached target tolerance {:.3f} with current tolerance {:.3f}".format(
                args['target_tolerance'], tol))
            break
        
        
        with ContextStopwatch() as train_model_stopwatch:  
            base_x_embedding_fn, base_x_embedding_dim = gendata.get_base_x_embedding(
                args['pretrained_featemb_model_path'] if args.get('use_pretrained_featemb', False) else "", 
                args.get('pretrained_featemb_use_logits', False))

            (model_fn, model_vars, model_vars_truehat, 
                small_base_x_embedding_train, small_base_x_embedding_targt, 
                small_y_train, xmins, new_x_nn, train_metrics) = train_pl_model( 
                    x=torch.from_numpy(samples_x), 
                    base_dm=gendata.dm,
                    search_domain=torch.from_numpy(gendata.search_domain), 
                    base_x_embedding_fn=base_x_embedding_fn, 
                    base_x_embedding_dim=base_x_embedding_dim,
                    train_y_savedir=gendata.log_dir,
                    do_normalize_y=args['do_normalize_output'],
                    output_ensemble_xmin=True,
                    noiseed=it,
                    **args
                )
            del base_x_embedding_fn, base_x_embedding_dim
            gc_cuda()
        
        ignore_flag_fxmin = ignore_criterion(samples_x, new_x_nn)
        ## Acquire xmin and adding to the observed samples   
        if (not ignore_flag_fxmin):
            ## Acquiring xmin 
            new_y_nn_train, new_y_nn_targt = gendata.evaluate_all_and_save_individual(
                np.array(new_x_nn).astype(dtype), args['n_initial_samples']+it)
            
            ## Add the latest acquired data points to the training set
            samples_x = np.vstack((samples_x, new_x_nn)).astype(dtype) 
            samples_y_train = np.hstack((samples_y_train, new_y_nn_train)).astype(dtype)
            samples_y_targt = np.hstack((samples_y_targt, new_y_nn_targt))
            
            it += 1

            ## Update min values
            # the i-th sample that gives the best loss value on dataset A 
            idx_obsmin = np.argmin(samples_y_train) # if (not ignore_flag) else np.argmin(samples_y)
            # the corresponding hyperparameters
            x_obsmin = samples_x[idx_obsmin] 
            # the best loss on dataset A   
            tol_train_obsmin = samples_y_train[idx_obsmin] - opt_target
            # the corresponding loss on dataset B
            tol_targt_obsmin = samples_y_targt[idx_obsmin] - opt_target
            
            print("Round {:d} acquires xmin({}): (x,y)=({:s},{:.3f}). Total training takes {:.3f}s.".format(
                it,  
                'fxmin',
                print_x(new_x_nn), 
                float(new_y_nn_train),
                train_model_stopwatch.elapsed_time
            )) 
            
            print_progress(
                nsample   = samples_x.shape[0],
                tol       = tol_targt_obsmin,  
                train_tol = tol_train_obsmin, 
                xmin      = x_obsmin
            ) 

            iterations.append(
                dict(
                    step=it, 
                    tol=tol_targt_obsmin,  
                    train_tol=tol_train_obsmin, 
                    xmin=x_obsmin.tolist(), 
                    scores='fxmin',
                    chosen_samples         = new_x_nn.tolist(), 
                    y_train_chosen_samples = new_y_nn_train.tolist(), 
                    y_targt_chosen_samples = new_y_nn_targt.tolist(), 
                    train_model_elapsed_time=0,
                    batch_acquisition_elapsed_time=0,
                )
            )  

            tol = copy.deepcopy(tol_targt_obsmin) 
            
            ## Retrain the model 
            with ContextStopwatch() as retrain_fxmin_stopwatch:
                base_x_embedding_fn, base_x_embedding_dim = gendata.get_base_x_embedding(
                    args['pretrained_featemb_model_path'] if args['use_pretrained_featemb'] else "", 
                    args.get('pretrained_featemb_use_logits', False))
                (model_fn, model_vars, model_vars_truehat, 
                 small_base_x_embedding_train, small_base_x_embedding_targt, 
                 small_y_train, xmins, _, train_metrics) = train_pl_model( 
                    x=torch.from_numpy(samples_x), 
                    base_dm=gendata.dm,
                    search_domain=torch.from_numpy(gendata.search_domain), 
                    base_x_embedding_fn=base_x_embedding_fn, 
                    base_x_embedding_dim=base_x_embedding_dim,
                    train_y_savedir=gendata.log_dir,
                    do_normalize_y=args['do_normalize_output'],
                    output_ensemble_xmin=False,
                    noiseed=it,
                    **args
                )
                del base_x_embedding_fn, base_x_embedding_dim
                gc_cuda()
            
        else: 
            print("Attempted to acquire xmin(NN)=({:s}) but ignore criterion met.".format(
                print_x(new_x_nn)))
        
        ## Acquiring new data points
        with ContextStopwatch() as batch_acquisition_stopwatch: 
            acquire_influence_max = InfluenceMax(
                available_x       = jnp.array(samples_x), 
                available_y       = small_y_train, 
                train_loss        = train_metrics[:args['n_candidate_model']], 
                xmins             = xmins,
                search_domain     = jnp.array(gendata.search_domain), 
                model_fn          = model_fn,
                base_x_embedding_train = small_base_x_embedding_train,
                base_x_embedding_targt = small_base_x_embedding_targt,
                model_vars        = model_vars, 
                model_vars_true   = model_vars_truehat, 
                acquire_k         = args['available_sample_k'], 
                m_kmeansplusplus  = args['m_kmeanspp'],
                **args
            )
            batch  = acquire_influence_max.compute_optima()
            
            new_x  = batch.samples
            scores = batch.scores
            
            ignore_flag = ignore_criterion(samples_x=samples_x, 
                                           new_x=new_x, 
                                           threshold=args.get('ignore_threshold', 0.01),
                                           search_domain=gendata.search_domain)
            if ignore_flag:  
                scores = np.full((args['available_sample_k'],), 'infmax-perturbed')
                while ignore_flag: 
                    new_x       = perturb(np.array(new_x), 
                                          threshold=args.get('ignore_threshold', 0.01),
                                          search_domain=gendata.search_domain)
                    ignore_flag = ignore_criterion(samples_x=samples_x, 
                                                   new_x=new_x, 
                                                   threshold=args.get('ignore_threshold', 0.01),
                                                   search_domain=gendata.search_domain)
        ## Observe new values
        new_y_train, new_y_targt = gendata.evaluate_all_and_save_individual(
            np.array(new_x).astype(dtype), 
            args['n_initial_samples']+it
        )
    
        ## Adding the latest acquired data points to the training set
        samples_x       = np.vstack((samples_x, new_x)).astype(dtype) 
        samples_y_train = np.hstack((samples_y_train, new_y_train)).astype(dtype)
        samples_y_targt = np.hstack((samples_y_targt, new_y_targt))
        
        scores_to_print = 'infmax-perturbed' if isinstance(scores[0], str) else '{:.3f}'.format(scores.mean())

        it += 1
        ## Update min values
        # the i-th sample that gives the best loss value on dataset A 
        idx_obsmin = np.argmin(samples_y_train) # if (not ignore_flag) else np.argmin(samples_y)
        # the corresponding hyperparameters
        x_obsmin   = samples_x[idx_obsmin]
        # the best loss on dataset A   
        tol_train_obsmin = samples_y_train[idx_obsmin] - opt_target
        # the corresponding loss on dataset B
        tol_targt_obsmin = samples_y_targt[idx_obsmin] - opt_target
         
        print("Round {:d} acquired: xnext: (x,y)=({:s},{:.3f}), scores={}. It takes {:.3f}s.".format(
            it, print_x(new_x), float(new_y_train), scores_to_print,
            batch_acquisition_stopwatch.elapsed_time+(retrain_fxmin_stopwatch.elapsed_time if not ignore_flag_fxmin else 0)
        ))
        
        print_progress(
            nsample   = samples_x.shape[0],
            tol       = tol_targt_obsmin,  
            train_tol = tol_train_obsmin, 
            xmin      = x_obsmin
        ) 
        
        iterations.append(
            dict(
                step=it, 
                tol=tol_targt_obsmin,  
                train_tol=tol_train_obsmin, 
                xmin=x_obsmin.tolist(), 
                scores=scores_to_print,
                chosen_samples           = new_x.tolist(), 
                y_train_chosen_samples   = new_y_train.tolist(), 
                y_targt_chosen_samples   = new_y_targt.tolist(), 
                train_model_elapsed_time = (train_model_stopwatch.elapsed_time
                                            +(retrain_fxmin_stopwatch.elapsed_time if not ignore_flag_fxmin else 0)),
                batch_acquisition_elapsed_time=batch_acquisition_stopwatch.elapsed_time
            )
        )  

        tol = copy.deepcopy(tol_targt_obsmin)
    print("DONE")
        
if __name__ == "__main__":
    torch.set_num_threads(1)
    main()


