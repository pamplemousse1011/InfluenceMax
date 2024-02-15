# Inspired by https://github.com/BlackHC/BatchBALD/tree/3cb37e9a8b433603fc267c0f1ef6ea3b202cfcb0
import argparse
import sys
import torch
from torch.utils.data import DataLoader
import numpy as np
import itertools
import functools
from blackhc import laaos


from context_stopwatch import ContextStopwatch
from data_generator import DatasetEnum, get_targets, get_experiment_data, update_data_loader
from random_fixed_length_sampler import RandomFixedLengthSampler
from utils import get_base_indices
from influence_max import InfluenceMax, AcquisitionBatch


def create_experiment_config_argparser(parser):
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=512, 
        help="input batch size for training"
    )
    
    parser.add_argument(
        "--scoring_batch_size", 
        type=int, 
        default=4096, 
        help="input batch size for scoring"
    )

    parser.add_argument(
        "--test_batch_size", 
        type=int, 
        default=16384, 
        help="input batch size for testing"
    )

    parser.add_argument(
        "--validation_set_size",
        type=int,
        default=1024,
        help="validation set size (0 for len(test_dataset) or whatever we got from the dataset)",
    )

    parser.add_argument(
        "--max_epochs", 
        type=int, 
        default=4, 
        help="maximum number of epochs to train"
    )

    parser.add_argument(
        "--min_epochs", 
        type=int, 
        default=1, 
        help="minimum number of epochs to train"
    )

    parser.add_argument(
        "--epoch_samples", 
        type=int, 
        default=101120,
        help="number of epochs to train"
    )
    
    parser.add_argument(
        "--available_sample_k",
        type=int,
        default=1,
        help="number of active samples to add per active learning iteration",
    )

    parser.add_argument(
        "--target_num_acquired_samples", 
        type=int, 
        default=3000, 
        help="max number of samples to acquire"
    )

    parser.add_argument(
        "--target_accuracy", 
        type=float, 
        default=0.80, 
        help="max accuracy to train to"
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
        "--num_gpus", 
        type=int, 
        default=1, 
        help="Number of gpus used for training"
    )

    parser.add_argument(
        "--seed", 
        type=int, 
        default=1, 
        help="random seed"
    )
    
    parser.add_argument(
        "--initial_samples_per_class",
        type=int,
        default=10,
        help="how many samples per class should be selected for the initial training set",
    )

    parser.add_argument(
        "--initial_sample",
        dest="initial_samples",
        type=int,
        action="append",
        help="sample that needs to be part of the initial samples (instead of sampling initial_samples_per_class)",
        default=None,
    )
    
    parser.add_argument(
        "--dataset",
        type=DatasetEnum,
        default=DatasetEnum.movie,
        help=f"dataset to use (options: {[f.name for f in DatasetEnum]})",
    )

    parser.add_argument(
        "--path_laaos_prefix",
        type=str,
        default = './activeLearning/laaos/',
        help="path of the prefix for laaos",
    )
    
    parser.add_argument(
        "--path_data",
        type=str,
        default = './activeLearning/data/',
        help="path of the MNIST dataset",
    )

    parser.add_argument(
        "--num_models",
        type=int,
        default=10,
        help="number of iterations/models to compute (jacknife/ensemble) estimator of Ey",
    )

    parser.add_argument(
        "--leave_one_out",
        action="store_true", 
        default=False, 
        help="whether we want to perform leave-one-out in estimation of Ey",
    )

    parser.add_argument(
        "--early_stopping",
        action="store_true", 
        default=False, 
        help="whether perform early_stopping",
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
        default=1, 
        help="one check happens after every _ numbers of training epochs",
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="learning rate used in optim.Adam for training",
    )

    parser.add_argument(
        "--T",
        type=int,
        default=1,
        help="number of independent runs whose results to be averaged to compute inverse hvp",
    )

    parser.add_argument(
        "--J",
        type=int,
        default=10,
        help="number of recursive computation so that approximation converges in inverse hvp computation",
    )

    parser.add_argument(
        "--progress_bar",
        action="store_true", 
        default=False, 
        help="whether print the progress",
    )
    return parser


def main():
    parser = argparse.ArgumentParser(
        description="InfMax", formatter_class=functools.partial(argparse.ArgumentDefaultsHelpFormatter, width=120)
    )
    parser.add_argument(
        "--experiment_task_id", 
        type=str, 
        default=None, 
        help="experiment id"
    )
    parser.add_argument(
        "--experiment_description", 
        type=str, 
        default="Type some stuff...", 
        help="Description of the experiment"
    )
    parser = create_experiment_config_argparser(parser)
    args = parser.parse_args()



    if args.dataset.name == "mnist":
        args.initial_samples = [38043, 40091, 17418,  2094, 39879,  3133,  5011, 
                                40683, 54379, 24287,  9849, 59305, 39508, 39356, 
                                 8758, 52579, 13655, 7636, 21562, 41329]
    elif args.dataset.name == "movie":
        args.initial_samples = [34375, 19226, 24558, 37524, 35193, 40918,  6929, 
                                35691, 41094, 25512, 29168, 18739, 40778, 45521, 
                                 2103,  3358, 39700, 27360,  9000, 43048]
    elif args.dataset.name == "tweet":
        args.initial_samples = [41857,  3110,  6855, 13971, 36915,  8597,  4412,
                                21388, 15319, 30267, 12734, 51750, 24997, 57826,
                                31700, 46833, 22118, 12260, 27604, 41120]

    if args.experiment_task_id:
        store_name = args.experiment_task_id
    else:
        store_name = (str(args.dataset.name)
                      + '_bsize' 
                      + str(args.available_sample_k) 
                      + '_lightning'
                      + '_' + str(args.acquisition_method)
                      + '_seed' + str(args.seed))
                      
    # Make sure we have a directory to store the results in, and we don't crash!
    store = laaos.create_file_store(
        store_name,
        suffix="",
        prefix=args.path_laaos_prefix,
        truncate=False,
        type_handlers=(laaos.StrEnumHandler(), laaos.ToReprHandler()),
    )
    store["args"] = args.__dict__
    store["cmdline"] = sys.argv[:]

    print("|".join(sys.argv))
    print(args.__dict__)


    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    num_gpus = args.num_gpus if use_cuda else 0
    kwargs_gpu = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    print(f"Using {device} for computations, number of gpus is {num_gpus}.")

    
    if (args.early_stopping or args.use_validation_set):
        use_validation_set = True
    else:
        use_validation_set = False
        
    dataset_class: DatasetEnum = args.dataset
    experiment_data = get_experiment_data(
        data_source=dataset_class.get_data_source(path_data=args.path_data), 
        num_classes=dataset_class.num_classes,
        initial_samples=args.initial_samples,
        initial_samples_per_class=args.initial_samples_per_class,
        validation_set_size=args.validation_set_size,
        use_validation_set=True
    )
    
    train_eval_loader = DataLoader(
        experiment_data.train_dataset,
        shuffle=True,
        batch_size=args.scoring_batch_size,
        **kwargs_gpu,
    )
    test_loader = DataLoader(
        experiment_data.test_dataset, 
        batch_size=args.test_batch_size, 
        shuffle=False, 
        **kwargs_gpu
    )
    available_loader_Ey = DataLoader(
        experiment_data.available_dataset, 
        batch_size=args.test_batch_size, 
        shuffle=False, 
        **kwargs_gpu
    )
    available_loader_hess = DataLoader(
        experiment_data.available_dataset, 
        batch_size=args.scoring_batch_size, 
        shuffle=False, 
        **kwargs_gpu
    )
    if use_validation_set:
        validation_loader = DataLoader(
            experiment_data.validation_dataset, 
            batch_size=args.test_batch_size, 
            shuffle=False, 
            **kwargs_gpu
        )
    else:
        validation_loader = None
 


    store["iterations"] = []
    # store wraps the empty list in a storable list, so we need to fetch it separately.
    iterations = store["iterations"]

    store["initial_samples"] = experiment_data.initial_samples


    for it in itertools.count(1):
        # we need to update the training dataloader every time due to the random sampling 
        # for jacknife leave-one-out
        
        train_loader = update_data_loader(
            dataset=experiment_data.train_dataset,
            sampler=RandomFixedLengthSampler(experiment_data.train_dataset, args.epoch_samples),
            batch_size=args.batch_size,
            num_models=args.num_models+1, 
            leave_one_out=args.leave_one_out,
            **kwargs_gpu
        )
    
        with ContextStopwatch() as train_model_stopwatch:
            model, test_metrics = dataset_class.train_model(
                args.num_models+1,
                train_loader,
                num_gpus,
                args,
                test_loader,
                validation_loader,
            )
        
       
        with ContextStopwatch() as batch_acquisition_stopwatch:
            if args.acquisition_method == "random":
                indices = experiment_data.active_learning_data.get_random_available_indices(args.available_sample_k)
                batch = AcquisitionBatch(indices, np.zeros_like(indices))
            elif args.acquisition_method == "infmax":
                acquire_influence_max = InfluenceMax(
                    model=model,
                    train_eval_loader=train_eval_loader,
                    available_loader_Ey=available_loader_Ey,
                    available_loader_hess=available_loader_hess,
                    test_loader=test_loader,
                    num_classes=dataset_class.num_classes,
                    args=args,
                    device=device,
                )
                batch = acquire_influence_max.compute_infmax_batch()
            
        
        original_batch_indices = get_base_indices(experiment_data.available_dataset, batch.indices)
        print(f"Acquiring indices {original_batch_indices}")
        targets = get_targets(experiment_data.available_dataset)
        acquired_targets = [int(targets[index]) for index in batch.indices]
        print(f"Acquiring targets {acquired_targets}")

        iterations.append(
            dict(
                acquisition_step=it,
                test_metrics=test_metrics,
                chosen_targets=acquired_targets,
                chosen_samples=original_batch_indices,
                chosen_samples_score=batch.scores,
                train_model_elapsed_time=train_model_stopwatch.elapsed_time,
                batch_acquisition_elapsed_time=batch_acquisition_stopwatch.elapsed_time,
            )
        )

        experiment_data.active_learning_data.acquire(batch.indices)
        print(f"Round{it}, acquicition_time {round(train_model_stopwatch.elapsed_time+batch_acquisition_stopwatch.elapsed_time)}")
        
        num_acquired_samples = len(experiment_data.active_learning_data.active_dataset) - len(
            experiment_data.initial_samples
        )
        if num_acquired_samples >= args.target_num_acquired_samples:
            print(f"{num_acquired_samples} acquired samples >= {args.target_num_acquired_samples}")
            break
        if test_metrics["acc"] >= args.target_accuracy:
            print(f'accuracy {test_metrics["acc"]} >= {args.target_accuracy}')
            break

    print("DONE")


if __name__ == "__main__":
    main()
