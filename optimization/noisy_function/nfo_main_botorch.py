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

from codes.context_stopwatch import ContextStopwatch
import codes.data_modules.data_generator as OptGenData
from codes.utils import print_x, print_progress
###################################################################

def create_experiment_config_argparser(parser):

    parser.add_argument(
        "--acquisition_method", 
        type=str, 
        default="ei", 
        help="acquisition method can be 'ei', 'skg', 'kg', 'ucb', 'mes'."
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
        default="Ackley",
        help=f"Ackley,Branin,Dropwave,GoldSteinPrice,Hartmann6,Rastr",
    )
    
    parser.add_argument(
        "--sample_var",
        type=float,
        default=0,
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
     
    if args.sample_var < 0:
        store_name = '_'.join([
            args.dataset,
            'h' + str(args.high_dim) +'l' + str(args.low_dim),
            'var' + str(args.noise_level),
            'bsize' + str(args.available_sample_k), 
            'seed' + str(args.seed),
            'trial'+ str(args.trial),
            '+'.join(['gp', args.acquisition_method]) 
        ])
    else:
        store_name = '_'.join([
            args.dataset,
            'h' + str(args.high_dim) +'l' + str(args.low_dim),
            'var' + str(args.sample_var),
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


    ## Current status
    opt_target  = gendata._f_opt
    idx_obsmin = np.argmin(samples_y)
    x_obsmin   = samples_x[idx_obsmin]
    tol_targt_obsmin   = gendata.evaluate_true(x_obsmin) - opt_target
    tol_train_obsmin = samples_y[idx_obsmin] - opt_target       

    print_progress(samples_x.shape[0], tol_targt_obsmin, tol_train_obsmin, x_obsmin)

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
    # args['pin_memory'] = True if args['use_cuda'] else False
    args['pin_memory'] = True 
    print(f"Using {device} for computations.")

    # Updating the number of initial sample points
    n_init = samples_x.shape[0]
    print(f"Starting from {n_init} number of samples, intend to acquire another {args['target_num_acquired_samples']}.")
    
    bounds = torch.stack([torch.from_numpy(gendata.search_domain[:,0]), 
                          torch.from_numpy(gendata.search_domain[:,1])]).to(device)
    n_dim = samples_x.shape[-1]
    
    tol = copy.deepcopy(tol_targt_obsmin)
    while (samples_x.shape[0] - args['n_initial_samples'] < args['target_num_acquired_samples']): 
        if tol <= args['target_tolerance']:
            print("Reached target tolerance {:.3f} with current tolerance {:.3f}".format(
                args['target_tolerance'], tol))
            break

        samples_x=samples_x.astype(dtype)
        samples_y=samples_y.astype(dtype)
        device=torch.device('cpu')

        ## Botorch selects new samples
        # botorch assumes maximization 
        with ContextStopwatch() as batch_acquisition_stopwatch:
            # Fit a Gaussian Process model to data 
            model = SingleTaskGP(torch.from_numpy(samples_x).to(device), 
                                 torch.from_numpy(-samples_y.reshape(-1,1)).to(device), 
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
                candidate_set = torch.from_numpy(gendata.gen_samples(1000).astype(dtype)).to(device)
                acq_func = qMaxValueEntropy(model, candidate_set)
                
            elif args['acquisition_method'] == 'ei':
                from botorch.acquisition import qExpectedImprovement
                acq_func = qExpectedImprovement(model, -samples_y.min())
            
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
            new_x = candidates.detach().cpu().numpy() 
            new_y = gendata.evaluate(np.array(new_x).astype(dtype))
            
        ## Observe new values
        samples_x = np.vstack((samples_x, new_x)).astype(dtype) 
        samples_y = np.hstack((samples_y, new_y)).astype(dtype)
        
        it += 1  
        ## Updating min values
        idx_obsmin = np.argmin(samples_y)
        x_obsmin   = samples_x[idx_obsmin]
        tol_targt_obsmin = gendata.evaluate_true(x_obsmin) - opt_target
        tol_train_obsmin = samples_y[idx_obsmin] - opt_target 

        print_progress(
            nsample   = samples_x.shape[0], 
            tol       = tol_targt_obsmin, 
            train_tol = tol_train_obsmin, 
            xmin      = x_obsmin
        )

        print("Round {:d} acquired: xnext: (x,y)=({:s},{:.3f}). It takes {:.3f}s".format(
            it, print_x(new_x), float(new_y[0]),  
            batch_acquisition_stopwatch.elapsed_time
        ))

        iterations.append(
            dict(
                step=it, 
                tol=tol_targt_obsmin, 
                train_tol=tol_train_obsmin, 
                xmin=x_obsmin.tolist(), 
                scores=scores.tolist(),
                chosen_samples=new_x.tolist(), 
                y_chosen_samples=new_y.tolist(),
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






