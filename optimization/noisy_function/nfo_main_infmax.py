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

from codes.utils import print_x, print_progress, ignore_criterion
from codes.context_stopwatch import ContextStopwatch
import codes.data_modules.data_generator as OptGenData 
from codes.influence_max.noisy_funct_optimization.influence_max import InfluenceMax  
from codes.influence_max.noisy_funct_optimization.opt_train_pl import train_pl_model

###################################################################

def create_experiment_config_argparser(parser):

    parser.add_argument(
        "--acquisition_method", 
        type=str, 
        default="infmax", 
        help="'infmax' or 'random'."
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
        default=5,
        help="number of initial active samples before the active optimization starts",
    )

    parser.add_argument(
        "--target_num_acquired_samples", 
        type=int, 
        default=300, 
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
        default=0.0001, 
        help="min tolerance to train to"
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
        help="Number of gpus used for training"
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
        default=1, 
        help="random seed"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="Ackley",
        help=f"Ackley,Branin,Dropwave,GoldSteinPrice,Hartmann6,Rastr",
    )
    
    parser.add_argument(
        "--sample_var",
        type=float,
        default=-1,
        help="var of the gaussian noise",
    )
    
    parser.add_argument(
        "--noise_level",
        type=float,
        default=0,
        help="if sample_var = -1.0, var of the gaussian noise is noise_level * var(y) \
              where y are randomly generated in the noise-free setting",
    )

    parser.add_argument(
        "--low_dim",
        type=int,
        default=4,
        help="dimensions to generate dataset using PPCA",
    )

    parser.add_argument(
        "--high_dim",
        type=int,
        default=0,
        help="dimensions of the inputs (default value is 0, meaning not in high-dimensional setting)",
    )

    parser.add_argument(
        "--path_laaos_prefix",
        type=str,
        default='./data-out/',
        # default='/home/',
        help="path of the prefix for laaos",
    )

    parser.add_argument(
        "--path_logs",
        type=str,
        default='./logs/',
        help="path of the lightning logs",
    ) 

    parser.add_argument(
        "--use_double", 
        action="store_true", 
        default=False, 
        help="whether to use double precision; if not, float"
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
        default=[512, 64, 64],
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
        "--ignore_threshold",
        type=float,
        default=1e-3,
        help="if any of the acquired samples has all dimensions with \
              its absolute difference of the candidate's corresponding \
              dimension smaller than the threshold, then we should ignore. \
              The default is we should not ignore at all.",
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
     
    if args.sample_var < 0:
        store_name = '_'.join([
            args.dataset,
            'h' + str(args.high_dim) +'l' + str(args.low_dim),
            'var' + str(args.noise_level),
            'bsize' + str(args.available_sample_k), 
            'seed' + str(args.seed),
            'trial'+ str(args.trial),
            args.acquisition_method 
        ])
    else:
        store_name = '_'.join([
            args.dataset,
            'h' + str(args.high_dim) +'l' + str(args.low_dim),
            'var' + str(args.sample_var),
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
        resume_file = laaos.safe_load(resume_filename)
        # if 'iterations' in resume_file and len(resume_file['iterations']) > 0:
        #     print("DONE.")
        #     return 
        print(f"Loading from {resume_filename}...")
        resume_args = resume_file['args']
        dtype = np.float64 if resume_args['use_double'] else np.float32

        if 'initial_samples' in resume_file and len(resume_file['initial_samples']) > 0:
            initial_to_generate = False  
            initial_samples = resume_file['initial_samples']
            samples_x=np.array(initial_samples['init_x'], dtype=dtype)
            samples_y=np.array(initial_samples['init_y'], dtype=dtype)
            
            gendata = getattr(OptGenData, resume_args['dataset'])(
                low_dim=resume_args['low_dim'],
                high_dim=resume_args['high_dim'],
                sample_var=resume_args['sample_var'],
                noise_level=resume_args['noise_level'],
                use_double=resume_args['use_double'])
            
            if 'iterations' in resume_file and len(resume_file['iterations']) > 0:
                initial_iterations = resume_file['iterations']
                for item in initial_iterations:
                    chosen_samples = np.array(item['chosen_samples'], dtype=dtype)
                    samples_x=np.vstack([samples_x, chosen_samples]) 
                    samples_y=np.hstack([samples_y, item['y_chosen_samples']]).astype(dtype)
                    
                lastitem = initial_iterations[-1]
                it = lastitem['step'] 

                if lastitem['step'] >= args.target_num_acquired_samples:
                    print("DONE.")
                    return
    
        resume_args['target_num_acquired_samples'] = args.target_num_acquired_samples
        args = resume_args
    else:
        print(f"Creating new file {store_name}... ")
        args = args.__dict__
        dtype = np.float64 if args['use_double'] else np.float32
    print(args)     

    if initial_to_generate:    
        gendata = getattr(OptGenData, args['dataset'])(
                low_dim=args['low_dim'],
                high_dim=args['high_dim'],
                sample_var=args['sample_var'],
                noise_level=args['noise_level'],
                use_double=args['use_double'],
                seed=args['seed'])
        print(f"Noise variance = {round(gendata.sample_var, 4)}")
        
        # generate initial samples
        samples_x, samples_y = gendata.gen_samples(
            args['n_initial_samples'], evaluate = True)
        initial_samples = {
            'init_x':samples_x.tolist(), 
            'init_y':samples_y.tolist(), 
        } 

    samples_x = samples_x.astype(dtype)
    samples_y = samples_y.astype(dtype) 

    store = create_file_store(
        store_name,
        suffix="",
        prefix=args['path_laaos_prefix'],
        truncate=False,
        type_handlers=(laaos.StrEnumHandler(), laaos.ToReprHandler()),
    )
    store["args"] = args
    store["initial_samples"] = initial_samples
    store["iterations"] = []
    # store wraps the empty list in a storable list, so we need to fetch it separately.
    iterations = store["iterations"]

    if it > 0: 
        for item in initial_iterations:
            iterations.append(item)
    
    ## Setting compute 
    args['use_cuda'] = not args['no_cuda'] and torch.cuda.is_available()
    device = torch.device("cuda" if args['use_cuda'] else "cpu")
    args['pin_memory'] = True if args['use_cuda'] else False
    print(f"Using {device} for computations.")

    # Updating the number of initial sample points
    n_init = samples_x.shape[0]
    print(f"Starting from {n_init} number of samples, intend to acquire another {args['target_num_acquired_samples']}.")
    
    ## Current status
    opt_target = gendata._f_opt
    idx_obsmin = np.argmin(samples_y)
    x_obsmin   = samples_x[idx_obsmin]
    tol_targt_obsmin = gendata.evaluate_true(x_obsmin) - opt_target
    tol_train_obsmin = samples_y[idx_obsmin] - opt_target       

    print_progress(it, tol_targt_obsmin, tol_train_obsmin, x_obsmin)

    tol = copy.deepcopy(tol_targt_obsmin)
    ignore_flag, ignore_flag_fxmin = True, True
    while (samples_x.shape[0] - args['n_initial_samples'] < args['target_num_acquired_samples']): 
        if tol <= args['target_tolerance']:
            print("Reached target tolerance {:.3f} with current tolerance {:.3f}".format(
                args['target_tolerance'], tol))
            break 

        with ContextStopwatch() as train_model_stopwatch:  
            (model_fn, model_params, xmins, new_x_nn, ensmodel_fn, train_metrics) = train_pl_model( 
                search_domain        = gendata.search_domain,
                train_features       = samples_x,
                train_labels         = samples_y, 
                do_normalize_y       = args.get('do_normalize_output', True),
                output_ensemble_xmin = True,
                noiseed              = it,
                **args
            )
            
        ignore_flag_fxmin = ignore_criterion(samples_x, new_x_nn)
        if (not ignore_flag_fxmin):        
            # acquire noisy observations of xmin and add to the observed samples
            new_y_nn = gendata.evaluate(np.array(new_x_nn))
                
            samples_x = np.vstack((samples_x, new_x_nn)) 
            samples_y = np.hstack((samples_y, new_y_nn))    
            
            it += 1
            # update min values
            idx_obsmin = np.argmin(samples_y)
            x_obsmin   = samples_x[idx_obsmin]
            tol_targt_obsmin = gendata.evaluate_true(x_obsmin) - opt_target
            tol_train_obsmin = samples_y[idx_obsmin] - opt_target  

            print("Round {:d} acquires xmin({}): (x,y)=({:s},{:.3f}). Total training takes {:.3f}s.".format(
                it,  
                'fxmin',
                print_x(new_x_nn), 
                float(new_y_nn),
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
                    chosen_samples=new_x_nn.tolist(), 
                    y_chosen_samples=new_y_nn.tolist(),  
                    train_model_elapsed_time=0,
                    batch_acquisition_elapsed_time=0,
                )
            ) 

            tol = copy.deepcopy(tol_targt_obsmin) 

            # retrain the model 
            with ContextStopwatch() as retrain_fxmin_stopwatch:
                (model_fn, model_params, xmins, _, ensmodel_fn, train_metrics) = train_pl_model( 
                    search_domain        = gendata.search_domain,
                    train_features       = samples_x,
                    train_labels         = samples_y,
                    do_normalize_y       = args.get('do_normalize_output', True),
                    output_ensemble_xmin = False,
                    noiseed              = it,
                    **args
                )

        else: 
            print("Attempted to acquire xmin(NN)=({:s}) but ignore criterion met.".format(
                print_x(new_x_nn)))

        # acquire new data points
        with ContextStopwatch() as batch_acquisition_stopwatch:
            if args['acquisition_method'] == "random":
                new_x = gendata.gen_samples(args['available_sample_k'], evaluate=False)   
                scores = np.full((args['available_sample_k'],), 'random')

            elif args['acquisition_method'] == "infmax":
                acquire_influence_max = InfluenceMax(
                    train_features    = jnp.array(samples_x),
                    train_labels      = jnp.array(samples_y),
                    train_loss        = train_metrics[:args['n_candidate_model']], 
                    xmins             = xmins,
                    search_domain     = jnp.array(gendata.search_domain), 
                    model_fn          = model_fn,
                    model_params      = model_params, 
                    ensmodel_fn       = ensmodel_fn,
                    acquire_k         = args['available_sample_k'], 
                    m_kmeansplusplus  = args['m_kmeanspp'],
                    **args
                )
                batch  = acquire_influence_max.compute_optima()
                
                new_x  = batch.samples
                scores = batch.scores 
            else:
                raise ValueError(f"Acquisition method not implemented, received {args['acquisition_method']}.")

            ignore_flag = ignore_criterion(samples_x, new_x)
            if ignore_flag: 
                scores = np.full((args['available_sample_k'],), 'random') 
                while ignore_flag:
                    new_x = gendata.gen_samples(args['available_sample_k'], evaluate = False)   
                    ignore_flag = ignore_criterion(samples_x, new_x)
        
        # observe new data points 
        new_y = gendata.evaluate(np.array(new_x)) 
        # add the latest acquired data points to the training set
        samples_x = np.vstack((samples_x, new_x))
        samples_y = np.hstack((samples_y, new_y))  
        scores_to_print = 'random' if isinstance(scores[0], str) else '{:.3f}'.format(scores.mean())
        
        it += 1
        # update min values
        idx_obsmin = np.argmin(samples_y)
        x_obsmin   = samples_x[idx_obsmin] 
        tol_targt_obsmin = gendata.evaluate_true(x_obsmin) - opt_target  
        tol_train_obsmin = samples_y[idx_obsmin] - opt_target  
        
        print("Round {:d} acquired: xnext: (x,y)=({:s},{:.3f}), scores={}. It takes {:.3f}s".format(
            it, print_x(new_x), float(new_y), scores_to_print,
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
                chosen_samples=new_x.tolist(), 
                y_chosen_samples=new_y.tolist(), 
                train_model_elapsed_time=(train_model_stopwatch.elapsed_time
                                          +(retrain_fxmin_stopwatch.elapsed_time if not ignore_flag_fxmin else 0)),
                batch_acquisition_elapsed_time=batch_acquisition_stopwatch.elapsed_time,
            )
        ) 

        tol = copy.deepcopy(tol_targt_obsmin)
    print("DONE")
        
if __name__ == "__main__":
    torch.set_num_threads(1)
    main()

