import os
import numpy as np
import torch


import warnings
import time
import argparse
import pickle5 as pickle


# import dependencies
from models import ActiveLearningInfluenceMax
from log_definition import Logger
from dataset_generator import GenSimDataset


def train_single_fin(
    B_true, 
    train_features, 
    train_labels, 
    test_features, 
    test_labels,
    alclass, 
    args = None
):
    acc      = torch.zeros(int(args.n_acquire*args.imax_bsize+2))
    labelled = np.zeros(int(args.n_acquire*args.imax_bsize+2))    
    
    mask = np.zeros(args.n_train, dtype = bool)
    ii = torch.randint(0, len( torch.where(train_labels.squeeze() == 1)[0]), [1])
    idx_new = torch.where(train_labels.squeeze() == 1)[0][ii]

    # note down the first index
    labelled[0]   = idx_new.numpy()
    mask[idx_new] = True

    ii = torch.randint(0, len( torch.where(train_labels.squeeze() == 0)[0]), [1])
    idx_new = torch.where(train_labels.squeeze() == 0)[0][ii]
    labelled[1] = idx_new.numpy()

    start_t = time.time()
    
    for t in range(args.n_acquire):
        # mask the new index
        mask[idx_new] = True

        if t == 0:
            # Obtain the current estimated parameter 
            B_curr, acc[1] = alclass.getCurrentEstParam(
                X = train_features[mask], 
                y = train_labels[mask],
                Xtest = test_features, 
                ytest = test_labels, 
                evaluation = True
            )
        else:
            # note down the index
            labelled[(2+(t-1)*args.imax_bsize-1) : (2+t*args.imax_bsize-1)] = idx_new
            # Obtain the current estimated parameter 
            B_curr, acc[(2+t*args.imax_bsize-1)] = alclass.getCurrentEstParam(
                X = train_features[mask], 
                y = train_labels[mask], 
                Xtest = test_features, 
                ytest = test_labels, 
                evaluation = True
            )

        if args.random_sampling:
            idx_new = np.random.choice(np.arange(args.n_train)[~mask], size=args.imax_bsize, replace=False)
        else:
            if not args.true_bias:
                if args.jacknife:
                    if sum(train_labels[mask] == 0) >= 2 and  sum(train_labels[mask] == 1) >= 2:
                        B_true_est = alclass.getLOOBiasEst(
                            X = train_features[mask], 
                            y = train_labels[mask]
                        )
                    else: 
                        B_true_est = B_curr - torch.ones_like(B_curr)
                else:
                    B_true_est = B_curr - torch.ones_like(B_curr)
            else: 
                B_true_est = B_true
        
        
       
            # Obtain the index for which the next data point to acquire
            idx_new = alclass.acquireNext(
                Xtest           = test_features, 
                Xtrain_acquired = train_features[mask], # if using optx then we need this
                mask            = mask,
                Xtrain          = train_features,
                B_curr          = B_curr, 
                B_true_est      = B_true_est
            )
            
        
        
        if t % 100 == 99:    # print every 200 acquired training data points
            elapsed_t = time.time() - start_t
            print("Acquire {}, Accuracy: {:.3f}, Time elapsed: {:<10.3f}".format(t*args.imax_bsize+1, 
                                                                                 acc[(2+t*args.imax_bsize-1)], 
                                                                                 elapsed_t))
            start_t = time.time()
        
        if t == args.n_acquire-1:
            # mask the new index
            mask[idx_new] = True

            # note down the index
            labelled[(2+t*args.imax_bsize-1) : (2+(t+1)*args.imax_bsize-1)] = idx_new
            # Obtain the current estimated parameter
            B_curr, acc[(2+(t+1)*args.imax_bsize-1)] = alclass.getCurrentEstParam(
                X = train_features[mask], 
                y = train_labels[mask],
                Xtest = test_features, 
                ytest = test_labels, 
                evaluation = True
            )
    

    
    acc      = acc.data.numpy()
    return acc, labelled

def train_single_cts(
    B_true, 
    train_features, 
    train_labels, 
    test_features, 
    test_labels,alclass, 
    args = None
):  
    acc = torch.zeros(int(args.n_acquire*args.imax_bsize+2))
    
    mask = np.zeros(args.n_train, dtype = bool)
    ii = torch.randint(0, len( torch.where(train_labels.squeeze() == 1)[0]), [1])
    # idx_new = torch.randint(0, n_train, [1])
    idx_new = torch.where(train_labels.squeeze() == 1)[0][ii]
    mask[idx_new] = True


    ii = torch.randint(0, len( torch.where(train_labels.squeeze() == 0)[0]), [1])
    idx_new = torch.where(train_labels.squeeze() == 0)[0][ii]
    mask[idx_new] = True


    start_t = time.time()
    
    train_features[:2, ] = train_features[mask]
    train_labels[:2, ]   = train_labels[mask]
    
    idx_curr = 2
    for t in range(args.n_acquire):
        if t >= 1:
            train_features[idx_curr:(idx_curr + args.imax_bsize),1:] = new_x
            train_labels[idx_curr:(idx_curr + args.imax_bsize)] = new_y
            idx_curr += args.imax_bsize

        if t == 0:
            # Obtain the current estimated parameter 
            B_curr, acc[1] = alclass.getCurrentEstParam(X = train_features[:idx_curr,:], 
                                                        y = train_labels[:idx_curr], 
                                                        Xtest = test_features, 
                                                        ytest = test_labels, 
                                                        evaluation = True)
        else:
            # Obtain the current estimated parameter 
            B_curr, acc[(2+t*args.imax_bsize-1)] = alclass.getCurrentEstParam(
                X = train_features[:idx_curr,:], 
                y = train_labels[:idx_curr],
                Xtest = test_features, 
                ytest = test_labels, 
                evaluation = True
            )

       
        if not args.true_bias:
            if args.jacknife:
                if (sum(train_labels[:idx_curr] == 0) >= 2 and 
                    sum(train_labels[:idx_curr] == 1) >= 2):
                    B_true_est = alclass.getLOOBiasEst(
                        X = train_features[:idx_curr,:], y = train_labels[:idx_curr]
                    )
                else: 
                    B_true_est = B_curr - torch.ones_like(B_curr)
            else:
                B_true_est = B_curr - torch.ones_like(B_curr)
        else: 
            B_true_est = B_true
    
    
    
        # Obtain the next data point to acquire
        new_x, new_y = alclass.acquireNext(
            Xtest           = test_features, 
            Xtrain_acquired = train_features[:idx_curr, :], 
            B_curr          = B_curr, 
            B_true_est      = B_true_est,
            B_true          = B_true
        )
        
        
        if t % 100 == 99:    # print every 200 acquired training data points
            elapsed_t = time.time() - start_t
            print("Acquire {}, Accuracy: {:.3f}, Time elapsed: {:<10.3f}".format(t*args.imax_bsize+2-1, 
                                                                                 acc[(2+t*args.imax_bsize-1)], 
                                                                                 elapsed_t))
            start_t = time.time()

        if t == args.n_acquire-1:
            train_features[idx_curr:(idx_curr + args.imax_bsize),1:] = new_x
            train_labels[idx_curr:(idx_curr + args.imax_bsize)] = new_y
            idx_curr += args.imax_bsize
            # Obtain the current estimated parameter
            B_curr, acc[(2+(t+1)*args.imax_bsize-1)] = alclass.getCurrentEstParam(
                X = train_features[:idx_curr,:], 
                y = train_labels[:idx_curr],
                Xtest = test_features,
                ytest = test_labels, 
                evaluation = True
            )

    acc      = acc.data.numpy()
    return acc 




def train_fin(gen, alclass, args):
    n = 0
    k = 0
    acc      = torch.zeros(args.n_iter, int(args.n_acquire*args.imax_bsize+2))
    labelled = np.zeros(args.n_iter, int(args.n_acquire*args.imax_bsize+2))
    while n < args.n_iter:
        torch.manual_seed(k)
        # Obtain the training dataset and the test set, with true training parameter B_true
        (B_true, train_features, train_labels, test_features, test_labels) = gen.load_simdata(seed=k)
        
        num1 = sum(train_labels) / len(train_labels)
        if num1 >= 0.3 and 1 - num1 >= 0.3:
            acc[n], labelled[n] = train_single_fin(
                B_true, 
                train_features, 
                train_labels, 
                test_features, 
                test_labels,
                alclass, 
                args
            )
            n += 1
        k += 1
    
    return acc, labelled

def train_cts(gen, alclass, args):
    n = 0
    k = 0
    acc = torch.zeros(args.n_iter, int(args.n_acquire*args.imax_bsize+2))
    while n < args.n_iter:
        torch.manual_seed(k)
        # Obtain the training dataset and the test set, with true training parameter B_true
        (B_true, train_features, train_labels, test_features, test_labels) = gen.load_simdata(seed=k)
        
        num1 = sum(train_labels) / len(train_labels)
        if num1 >= 0.3 and 1 - num1 >= 0.3:
            acc[n] = train_single_cts(
                B_true, 
                train_features, 
                train_labels, 
                test_features, 
                test_labels,
                alclass, 
                args
            )
            n += 1
        k += 1
    
    return acc


def train_single(args, gen, alclass):
    if args.finite:
        return train_fin(gen, alclass, args)
    else:
        return train_cts(gen, alclass, args)

def read_args_commandline():
    parser = argparse.ArgumentParser()
    # Parser for command-line options, arguments and subcommands
    # The argparse module makes it easy to write user-friendly command-line interfaces.
    
    ###############################################################################
    #                             General Settings                                #
    ###############################################################################
    parser.add_argument('--covtype', nargs='?',       const = 1, type = str,
                        default = "PPCA")         
    parser.add_argument('--imax_bsize', nargs='?',    const = 1, type = int,
                        default = int(1))    
    parser.add_argument('--optx', 
                        action = 'store_true')
    parser.add_argument('--finite', 
                        action = 'store_true')   
    parser.add_argument('--n_iter', nargs='?',        const = 1, type = int,
                        default = int(200))                         
    parser.add_argument('--n_train', nargs='?',       const = 1, type = int,
                        default = int(50000))
    parser.add_argument('--n_test', nargs='?',        const = 1, type = int,
                        default = int(5000))
    parser.add_argument('--path_logger', nargs='?',   const = 1, type = str,
                        default = '')
    parser.add_argument('--path_output', nargs='?',   const = 1, type = str, 
                        default = '')
    parser.add_argument('--input_dim', nargs = '?',   const = 1, type = int, 
                        default = int(20))
    parser.add_argument('--n_acquire', nargs = '?',   const = 1, type = int, 
                        default = int(500))
    parser.add_argument('--true_bias', 
                        action = 'store_true')
    parser.add_argument('--jacknife', 
                        action = 'store_true')
    parser.add_argument('--random_sampling',
                        action = 'store_true')

    return parser.parse_args()


def compile_filename(args):

    filename = ('sim_' + str(args.covtype)
                + '_nvar' + str(args.input_dim) 
                + '_ntrain' + str(args.n_train) 
                + '_ntest' + str(args.n_test) 
                + '_bsize'+ str(args.imax_bsize) + '_' 
                + '.pickle')
    return filename

def main():
    args = read_args_commandline()
    
    logger = Logger(args.path_logger)
    logger.write_settings(args)

    
    torch.backends.cudnn.enabled = False
    
    #### Finite case - remain
    args.finite      = True
    args.optx        = False
    ## want to run one first with true_bias
    args.true_bias = True
    args.jacknife  = False 
    gen = GenSimDataset(args)
    alclass         = ActiveLearningInfluenceMax(args)
    acc_tFbT_remain, labelled_tFbT_remain = train_single(gen, alclass, args)

    ## want to then run one also the jacknife
    args.jacknife = True 
    args.true_bias = False
    gen = GenSimDataset(args)
    alclass           = ActiveLearningInfluenceMax(args)
    acc_tFbFjT_remain, labelled_tFbFjT_remain = train_single(gen, alclass, args)
    
    ## want to then run one with one bias
    args.jacknife  = False
    args.true_bias = False
    gen = GenSimDataset(args)
    alclass           = ActiveLearningInfluenceMax(args)
    acc_tFbFjF_remain, labelled_tFbFjF_remain = train_single(gen, alclass, args)

    ## run one with random_sampling, but only with remain
    args.random_sampling = True
    gen = GenSimDataset(args)
    alclass              = ActiveLearningInfluenceMax(args)
    acc_random_remain, labelled_random_remain = train_single(gen, alclass, args)
    

    res = {
        'acc_tFbT_remain'       : acc_tFbT_remain,
        'acc_tFbFjT_remain'     : acc_tFbFjT_remain,
        'acc_tFbFjF_remain'     : acc_tFbFjF_remain,
        'acc_random_remain'     : acc_random_remain,
        'labelled_tFbT_remain'  : labelled_tFbT_remain,
        'labelled_tFbFjT_remain': labelled_tFbFjT_remain,
        'labelled_tFbFjF_remain': labelled_tFbFjF_remain,
        'labelled_random_remain': labelled_random_remain
    }
    filename = compile_filename(args)
    print ('Saving acc, acquired...' + filename)
    path_plus_name = os.path.join(args.path_output, filename)
    with open(path_plus_name, 'wb') as f:
        pickle.dump(res, f)

    

if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()


