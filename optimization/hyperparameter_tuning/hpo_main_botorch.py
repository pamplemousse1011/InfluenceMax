import argparse  
import warnings
import functools 
import os 
import copy

from blackhc.laaos import create_file_store
import laaos 

import numpy as np
import torch 

from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.models.transforms import Normalize, Standardize

from codes.utils import print_x, print_progress
from codes.context_stopwatch import ContextStopwatch
import codes.data_modules.data_generator as OptGenData
###################################################################

def create_experiment_config_argparser(parser):

    parser.add_argument(
        "--acquisition_method", 
        type=str, 
        default="ei", 
        help="acquisition method can be 'ei', 'skg', 'kg', 'ucb'."
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
        default=0, 
        help="min tolerance to train to"
    ) 

    parser.add_argument(
        "--no_cuda", 
        action="store_true", 
        default=False, 
        help="disables CUDA training"
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
        default="CIFAR10",
        help=f"CIFAR10",
    )

    parser.add_argument(
        "--net_type",
        type=str,
        default="preactrn",
        help=f"vgg,lenet,cnn,resnet,wide_resnet",
    )

    parser.add_argument(
        "--path_laaos_prefix",
        type=str,
        default='/scratch/wy635/active_optimization/singularity/data-out/',
        # default='/home/',
        help="path of the prefix for laaos",
    )

    parser.add_argument(
        "--path_logs",
        type=str,
        default='/scratch/wy635/active_optimization/singularity/data-out/',
        help="path of the lightning logs",
    )
    
    parser.add_argument(
        "--path_data",
        type=str,
        default='/scratch/wy635/data/',
        help="path of the MNIST dataset",
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
        "--n_devices", 
        type=int, 
        default=1, 
        help="Number of devices used in trainer"
    )

    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=0, 
        help="Number of workers used in dataloader"
    )
    
    return parser

NUM_FANTASIES=128
BETA=0.1
NUM_RESTARTS=10  
RAW_SAMPLES=512  

def main():
    parser = argparse.ArgumentParser(
        description="BoTorch", formatter_class=functools.partial(argparse.ArgumentDefaultsHelpFormatter, width=120)
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
        '+'.join(['gp', args.acquisition_method]) 
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
            samples_y_avail = np.array(initial_samples['init_y_train'], dtype=dtype)
            samples_y_targt = np.array(initial_samples['init_y_targt'], dtype=dtype) 
            
            gendata = getattr(OptGenData, args['dataset'])(  
                use_double=args['use_double'],
                net_type=args['net_type'],
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
                    chosen_samples = np.array(item['chosen_samples'], dtype=dtype) 
                    samples_x       = np.vstack([samples_x, chosen_samples]) 
                    samples_y_avail = np.hstack([samples_y_avail, item['y_train_chosen_samples']]).astype(dtype)
                    samples_y_targt = np.hstack([samples_y_targt, item['y_targt_chosen_samples']]).astype(dtype)
                    
                lastitem = initial_iterations[-1]
                it = lastitem['step'] 

                if lastitem['step'] >= args['target_num_acquired_samples']:
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
            use_double=args['use_double'],
            net_type=args['net_type'], 
            data_dir=args['path_data'],
            log_dir=args['path_logs'],
            seed=args['seed'],
            shift=args.get('covariate_shift', False),
            in_distribution=args.get('shift_in_distribution', False),
            num_workers=args['num_workers'],
            pin_memory=args['pin_memory'],
            n_devices=args['n_devices']
        )
        
        # generate initial samples
        samples_x = gendata.gen_samples(args['n_initial_samples'])

        samples_y_avail = []
        samples_y_targt = [] 
        for ii in range(args['n_initial_samples']):
            # call the base model 
            out_avail, out_targt = gendata.evaluate(samples_x[ii])
            samples_y_avail.append(out_avail)  
            samples_y_targt.append(out_targt) 

        samples_y_avail = np.hstack(samples_y_avail).astype(samples_x.dtype)
        samples_y_targt = np.hstack(samples_y_targt).astype(samples_x.dtype) 
            
        initial_samples = {'init_x'  : samples_x.tolist(), 
                           'init_y_train': samples_y_avail.tolist(), 
                           'init_y_targt': samples_y_targt.tolist()}  

    ## Current status
    opt_target = gendata._f_opt
    # the i-th sample that gives the best loss value on dataset A 
    idx_obsmin = np.argmin(samples_y_avail)
    # the corresponding hyperparameters
    x_obsmin   = samples_x[idx_obsmin]
    # the best loss on dataset A   
    tol_train_obsmin = samples_y_avail[idx_obsmin] - opt_target
    # the corresponding loss on dataset B
    tol_targt_obsmin = samples_y_targt[idx_obsmin] - opt_target
    
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
    store["iterations"] = []
    # store wraps the empty list in a storable list, so we need to fetch it separately.
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
                y_train_chosen_samples = samples_y_avail[-args['available_sample_k']:].tolist(), 
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
    
    bounds = torch.stack([torch.from_numpy(gendata.search_domain[:,0]), 
                          torch.from_numpy(gendata.search_domain[:,1])]).to(device)
    n_dim = samples_x.shape[-1]

    while (samples_x.shape[0] - args['n_initial_samples'] < args['target_num_acquired_samples']): 
        if tol <= args['target_tolerance']:
            print("Reached target tolerance {:.3f} with current tolerance {:.3f}".format(
                args['target_tolerance'], tol))
            break
        # botorch assumes maximization 
        with ContextStopwatch() as batch_acquisition_stopwatch:
            # Fit a Gaussian Process model to data
            model = SingleTaskGP(torch.from_numpy(samples_x).to(device), 
                                 torch.from_numpy(-samples_y_avail.reshape(-1,1)).to(device), 
                                 outcome_transform=Standardize(m=1),
                                 input_transform=Normalize(d=n_dim, bounds=bounds))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            
            if args['acquisition_method'] == 'ucb':
                # Construct an acquisition function
                from botorch.acquisition import UpperConfidenceBound
                # Construct an acquisition function
                acq_func = UpperConfidenceBound(model, beta=BETA)
                
            elif args['acquisition_method'] == 'mes':
                from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy
                candidate_set = torch.from_numpy(
                    gendata.gen_samples(
                        1000,  
                        it+args['trial']
                    ).astype(dtype)
                ).to(device)
                acq_func = qMaxValueEntropy(model, candidate_set)
                
            elif args['acquisition_method'] == 'ei':
                from botorch.acquisition import qExpectedImprovement
                acq_func = qExpectedImprovement(model, -samples_y_avail.min())
            
            elif args['acquisition_method'] == 'skg':
                from botorch.acquisition import qKnowledgeGradient
                acq_func = qKnowledgeGradient(model, num_fantasies=NUM_FANTASIES)
            elif args['acquisition_method'] == 'kg':
                from botorch.acquisition import qKnowledgeGradient
                from botorch.acquisition import PosteriorMean
                qKG = qKnowledgeGradient(model, num_fantasies=NUM_FANTASIES)
                ## find the maximum posterior mean 
                ## - we can use a large number of random restarts and raw_samples 
                ## to increase the likelihood that we do indeed find it 
                ## (this is a non-convex optimization problem, after all).
                _, max_pmean = optimize_acqf(
                    acq_function=PosteriorMean(model),
                    bounds=bounds,
                    q=args['available_sample_k'],
                    num_restarts=NUM_RESTARTS,
                    raw_samples=RAW_SAMPLES,
                )
                ## optimize KG after passing the current value.
                acq_func = qKnowledgeGradient(
                    model,
                    num_fantasies=NUM_FANTASIES,
                    sampler=qKG.sampler,
                    current_value=max_pmean,
                )
            else:
                raise ValueError(f"Acquisition functions only implemented for 'ei', 'kg',\
                                 'ucb', 'mes'. Received {args['acquisition_method']}. ")
             
            # Optimize the acquisition function
            candidates, acq_values = optimize_acqf(
                acq_func, 
                bounds=bounds, 
                q=args['available_sample_k'],
                num_restarts=NUM_RESTARTS, # in optimize_posterior_samples, number of times to run gen_candidates_torch and pick the best
                raw_samples=RAW_SAMPLES, # in optimize_posterior_samples, pick num_restarts out of raw_samples number of samples generated from SobolEngine 
            )
            
            scores = acq_values.detach().cpu().numpy() 
            new_x  = candidates.detach().cpu().numpy() 

        # observe new values  
        new_y_avail, new_y_targt = gendata.evaluate(new_x.astype(dtype))
        
        samples_x = np.vstack((samples_x, new_x)).astype(dtype) 
        samples_y_avail = np.hstack((samples_y_avail, new_y_avail)).astype(dtype)
        samples_y_targt = np.hstack((samples_y_targt, new_y_targt))
            
        it += 1  
        # the i-th sample that gives the best loss value on dataset A 
        idx_obsmin = np.argmin(samples_y_avail) # if (not ignore_flag) else np.argmin(samples_y)
        # the corresponding hyperparameters
        x_obsmin = samples_x[idx_obsmin]
        # the best loss on dataset A   
        tol_train_obsmin = samples_y_avail[idx_obsmin] - opt_target
        # the corresponding loss on dataset B
        tol_targt_obsmin = samples_y_targt[idx_obsmin] - opt_target
        
        print("Round {:d} acquired: xnext: (x,y)=({:s},{:.3f}). It takes {:.3f}s".format(
            it, print_x(new_x), float(new_y_avail),  
            batch_acquisition_stopwatch.elapsed_time
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
                scores=scores.tolist(),
                chosen_samples           = new_x.tolist(), 
                y_train_chosen_samples   = new_y_avail.tolist(), 
                y_targt_chosen_samples   = new_y_targt.tolist(), 
                batch_acquisition_elapsed_time=batch_acquisition_stopwatch.elapsed_time,
            )
        )  

        tol = copy.deepcopy(tol_targt_obsmin)

    print("DONE")
        

 
      

if __name__ == "__main__":
    torch.set_num_threads(1) 
    warnings.filterwarnings("ignore", category=RuntimeWarning) 
    warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning) 
    main()




