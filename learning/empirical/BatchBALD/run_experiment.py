import argparse
import sys
import torch

from acquisition_method import AcquisitionMethod
from context_stopwatch import ContextStopwatch
from dataset_enum import DatasetEnum, get_targets, get_experiment_data
from random_fixed_length_sampler import RandomFixedLengthSampler
from torch_utils import get_base_indices
import torch.utils.data as data

from acquisition_functions import AcquisitionFunction

from blackhc import laaos

# NOTE(blackhc): get the directory right (oh well)
import blackhc.notebook

import functools
import itertools

def create_experiment_config_argparser(parser):
    parser.add_argument("--batch_size", type=int, default=512, help="input batch size for training")
    parser.add_argument("--scoring_batch_size", type=int, default=4096, help="input batch size for scoring")
    parser.add_argument("--test_batch_size", type=int, default=16384, help="input batch size for testing")
    parser.add_argument(
        "--validation_set_size",
        type=int,
        default=1024,
        help="validation set size (0 for len(test_dataset) or whatever we got from the dataset)",
    )
    parser.add_argument(
        "--early_stopping_patience", type=int, default=3, help="# patience epochs for early stopping per iteration"
    )
    parser.add_argument("--epochs", type=int, default=2, help="number of epochs to train")
    parser.add_argument("--epoch_samples", type=int, default=101120, help="number of epochs to train")
    parser.add_argument("--num_inference_samples", type=int, default=50, help="number of samples for inference")
    parser.add_argument(
        "--available_sample_k",
        type=int,
        default=1,
        help="number of active samples to add per active learning iteration",
    )
    parser.add_argument("--target_num_acquired_samples", type=int, default=2000, help="max number of samples to acquire")
    parser.add_argument("--target_accuracy", type=float, default=0.74, help="max accuracy to train to")
    parser.add_argument("--no_cuda", action="store_true", default=False, help="disables CUDA training")
    parser.add_argument("--quickquick", action="store_true", default=False, help="uses a very reduced dataset")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--log_interval", type=int, default=20, help="how many batches to wait before logging training status"
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
        "--type",
        type=AcquisitionFunction,
        default=AcquisitionFunction.bald,
        help=f"acquisition function to use (options: {[f.name for f in AcquisitionFunction]})",
    )
    parser.add_argument(
        "--acquisition_method",
        type=AcquisitionMethod,
        default=AcquisitionMethod.multibald,
        help=f"acquisition method to use (options: {[f.name for f in AcquisitionMethod]})",
    )
    parser.add_argument(
        "--dataset",
        type=DatasetEnum,
        default=DatasetEnum.tweet,
        help=f"dataset to use (options: {[f.name for f in DatasetEnum]})",
    )
    parser.add_argument(
        "--min_remaining_percentage",
        type=int,
        default=100,
        help="how much of the available dataset should remain after culling in BatchBALD",
    )
    parser.add_argument(
        "--min_candidates_per_acquired_item",
        type=int,
        default=20,
        help="at least min_candidates_per_acquired_item*acqusition_size should remain after culling in BatchBALD",
    )
    parser.add_argument(
        "--initial_percentage",
        type=int,
        default=100,
        help="how much of the available dataset should be kept before scoring (cull randomly for big datasets)",
    )
    parser.add_argument(
        "--reduce_percentage",
        type=int,
        default=0,
        help="how much of the available dataset should be culled after each iteration",
    )
    parser.add_argument(
        "--balanced_validation_set",
        action="store_true",
        default=False,
        help="uses a balanced validation set (instead of randomly picked)"
        "(and if no validation set is provided by the dataset)",
    )
    parser.add_argument(
        "--balanced_test_set",
        action="store_true",
        default=False,
        help="force balances the test set---use with CAUTION!",
    )
    parser.add_argument(
        "--path_data",
        type=str,
        default = './data/',
        help="path of the MNIST dataset",
    )
    parser.add_argument(
        "--path_laaos_prefix",
        type=str,
        default = './laaos/',
        help="path of the prefix for laaos",
    )
    return parser


def main():
    parser = argparse.ArgumentParser(
        description="BatchBALD", formatter_class=functools.partial(argparse.ArgumentDefaultsHelpFormatter, width=120)
    )
    parser.add_argument("--experiment_task_id", type=str, default=None, help="experiment id")
    parser.add_argument(
        "--experiments_laaos", type=str, default=None, help="Laaos file that contains all experiment task configs"
    )
    parser.add_argument(
        "--experiment_description", type=str, default="Trying stuff..", help="Description of the experiment"
    )
    parser = create_experiment_config_argparser(parser)
    args = parser.parse_args()

    if args.experiments_laaos is not None:
        config = laaos.safe_load(
            args.experiments_laaos, expose_symbols=(AcquisitionFunction, AcquisitionMethod, DatasetEnum)
        )
        # Merge the experiment config with args.
        # Args take priority.
        args = parser.parse_args(namespace=argparse.Namespace(**config[args.experiment_task_id]))

    # DONT TRUNCATE LOG FILES EVER AGAIN!!! (OFC THIS HAD TO HAPPEN AND BE PAINFUL)
    reduced_dataset = args.quickquick

    store_name = (str(args.dataset.name)
                    + '_bsize' 
                    + str(args.available_sample_k) 
                    + '_igniteversion'
                    + '_' + str(args.acquisition_method.name) 
                    + '_seed' + str(args.seed))
                      
       
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
    
    # Make sure we have a directory to store the results in, and we don't crash!
    store = laaos.create_file_store(
        store_name,
        suffix="",
        prefix=args.path_laaos_prefix,
        truncate=False,
        type_handlers=(blackhc.laaos.StrEnumHandler(), blackhc.laaos.ToReprHandler()),
    )
    store["args"] = args.__dict__
    store["cmdline"] = sys.argv[:]

    print("|".join(sys.argv))
    print(args.__dict__)

    acquisition_method: AcquisitionMethod = args.acquisition_method

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    print(f"Using {device} for computations")

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    dataset: DatasetEnum = args.dataset
    samples_per_class = args.initial_samples_per_class
    validation_set_size = args.validation_set_size
    balanced_test_set = args.balanced_test_set
    balanced_validation_set = args.balanced_validation_set

    experiment_data = get_experiment_data(
        data_source=dataset.get_data_source(args.path_data),
        num_classes=dataset.num_classes,
        initial_samples=args.initial_samples,
        reduced_dataset=reduced_dataset,
        samples_per_class=samples_per_class,
        validation_set_size=validation_set_size,
        balanced_test_set=balanced_test_set,
        balanced_validation_set=balanced_validation_set,
    )

    test_loader = torch.utils.data.DataLoader(
        experiment_data.test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs
    )

    train_loader = torch.utils.data.DataLoader(
        experiment_data.train_dataset,
        sampler=RandomFixedLengthSampler(experiment_data.train_dataset, args.epoch_samples),
        batch_size=args.batch_size,
        **kwargs,
    )
    train_eval_loader = torch.utils.data.DataLoader(
        experiment_data.train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        **kwargs,
    )
    available_loader = torch.utils.data.DataLoader(
        experiment_data.available_dataset, batch_size=args.scoring_batch_size, shuffle=False, **kwargs
    )

    validation_loader = torch.utils.data.DataLoader(
        experiment_data.validation_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs
    )

    store["iterations"] = []
    # store wraps the empty list in a storable list, so we need to fetch it separately.
    iterations = store["iterations"]

    store["initial_samples"] = experiment_data.initial_samples

    acquisition_function: AcquisitionFunction = args.type
    max_epochs = args.epochs
    
    if args.acquisition_method.name == "random":
        bayes = False
    else:
        bayes = True

    for it in itertools.count(1):

        def desc(name):
            return lambda engine: "%s: %s (%s samples)" % (name, it, len(experiment_data.train_dataset))

        with ContextStopwatch() as train_model_stopwatch:
            early_stopping_patience = args.early_stopping_patience
            num_inference_samples = args.num_inference_samples
            log_interval = args.log_interval

            model, num_epochs, test_metrics = dataset.train_model(
                train_loader,
                test_loader,
                validation_loader,
                num_inference_samples,
                max_epochs,
                early_stopping_patience,
                desc,
                log_interval,
                device,
                bayes = bayes
            )

        with ContextStopwatch() as batch_acquisition_stopwatch:
            batch = acquisition_method.acquire_batch(
                model=model,
                acquisition_function=acquisition_function,
                available_loader=available_loader,
                num_classes=dataset.num_classes,
                k=args.num_inference_samples,
                b=args.available_sample_k,
                min_candidates_per_acquired_item=args.min_candidates_per_acquired_item,
                min_remaining_percentage=args.min_remaining_percentage,
                initial_percentage=args.initial_percentage,
                reduce_percentage=args.reduce_percentage,
                device=device,
                dat_train_model=dataset.train_model,
                train_loader=train_eval_loader,
                validation_loader=validation_loader,
                test_loader=test_loader,
                args=args,
                kwargs=kwargs,
                desc=desc
            )

        original_batch_indices = get_base_indices(experiment_data.available_dataset, batch.indices)
        print(f"Acquiring indices {original_batch_indices}")
        targets = get_targets(experiment_data.available_dataset)
        acquired_targets = [int(targets[index]) for index in batch.indices]
        print(f"Acquiring targets {acquired_targets}")

        iterations.append(
            dict(
                acquisition_step=it,
                num_epochs=num_epochs,
                test_metrics=test_metrics,
                chosen_targets=acquired_targets,
                chosen_samples=original_batch_indices,
                chosen_samples_score=batch.scores,
                chosen_samples_orignal_score=batch.orignal_scores,
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
        if test_metrics["accuracy"] >= args.target_accuracy:
            print(f'accuracy {test_metrics["accuracy"]} >= {args.target_accuracy}')
            break

    print("DONE")


if __name__ == "__main__":
    main()
