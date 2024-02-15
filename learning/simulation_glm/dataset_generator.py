import torch
# Import tensor dataset & data loader
import numpy as np
from scipy.stats import multivariate_normal as mvn
import scipy.linalg


class ExponentialFamilies(object):
    def __init__(self, X = None, y = None, param_true = None, expected = True, test = True):

        self.X          = X
        self.y          = y
        self.param_true = param_true
        self.expected   = expected
        self.test       = test
    
    def getEta(self, param):
        return torch.matmul(self.X, param)

    def getGradLoss(self, param):
        eta_curr = self.getEta(param)
        
        if self.expected:
            eta_true = self.getEta(self.param_true)
            out = (- torch.sigmoid(eta_true) + torch.sigmoid(eta_curr)) * self.X # n x (d + 1)
            return out.sum(dim = 0) if self.test else out 
        else:
            out = (-self.y + torch.sigmoid(eta_curr)) * self.X
            return out.sum(dim = 0) if self.test else out 


# dataset definition
class GenSimDataset(object):
    """
    Have the training data reshuffled at every epoch, but not test set
    No batch for validation set
    """
    # load the dataset
    def __init__(self, args=None):
        if args == None:
            self.input_dim = 20
            self.n_train   = 50000 
            self.n_test    = 5000
            self.covtype   = "PPCA"  
        else:
            self.input_dim   = args.input_dim
            self.n_train     = args._train
            self.n_test      = args.n_test
            self.covtype     = args.covtype
        
    def simdata_model(self, n, variation = "PPCA", B = None):
        if variation == "PPCA":
            # low-rank perturbation cov(X) = theta @ theta.T + sigma^2 I  (sigma = 1 in this case)
            U = np.random.normal(size = (n, 3))
            theta = np.random.uniform(size = (3, self.input_dim))
            Xmean = U.dot(theta)
            X = Xmean + np.random.normal(np.zeros_like(Xmean)) * 0.1
            # check eigenvalue of the covariance:
            # np.linalg.eigvalsh(X.T @X / n)
            # If we compute np.linalg.eigvalsh(X.T @X / n) - 1, we will see two big eigen values. and others are nearly zero.
        elif variation == "high-corr":
            cov = scipy.linalg.toeplitz(0.75 ** np.arange(self.input_dim)) # make sure positive semidefinite
            # check eigen values:
            # np.linalg.eigvalsh(cov)
            X = mvn.rvs(mean = np.zeros(self.input_dim), cov=cov, size = n)
            # Sanity check (from covariance matrix to simple correlations): 
            # np.corrcoef(scores.T)
            # check https://scipy-cookbook.readthedocs.io/items/CorrelatedRandomSamples.html#:~:text=To%20generate%20correlated%20normally%20distributed,eigenvalues%20and%20eigenvectors%20of%20R.
        elif variation == "indep-norm":
            X = np.random.normal(size = (n, self.input_dim)) 
        elif variation == "indep-unif":
            X = np.random.uniform(size = (n, self.input_dim))
        elif variation == "bimodal":
            value_bern = np.random.binomial(1, 0.4, size = (n))
            value_norm1 = np.random.normal(0.1, 0.15, size = (n, self.input_dim)) 
            value_norm2 = np.random.normal(0.9, 0.15, size = (n, self.input_dim)) 
            X = value_bern * value_norm1 + (1-value_bern) * value_norm2
        else:
            raise ValueError('variation can only be "PPCA", "high-corr" or "indep-unif" or "indep-unif" or "bimodal". Received: {}.'.format(variation))
   
    
        if B is None:
            # We want a more balanced one:
            B = np.random.normal(0, 1, size = (self.input_dim + 1,))
        
        g = np.matmul(X, B[1:]) + B[0]
        # Noisy mean
        mu = 1 / (1 + np.exp(-g)) 
        # bernoulli response variable
        y = np.random.binomial(1, mu, size = (n,))
        
        return B, X, y

        
    def load_simdata(self, seed):
        np.random.seed(seed)
        param_true, X_train, y_train = self.simdata_model(n = self.n_train, variation = self.covtype)
        if self.covtype == "high-corr":
            _, X_test, y_test = self.simdata_model(n = self.n_test, B = param_true, variation = "indep-norm")
        elif self.covtype == "bimodal":
            _, X_test, y_test = self.simdata_model(n = self.n_test, B = param_true, variation = "indep-unif")
        else:
            _, X_test, y_test = self.simdata_model(n = self.n_test, B = param_true, variation = self.covtype)

        X_train    = torch.from_numpy(X_train) 
        y_train    = torch.from_numpy(y_train)
        
        X_test = torch.from_numpy(X_test) 
        y_test = torch.from_numpy(y_test)

        if param_true is not None:    
            param_true = torch.from_numpy(param_true)
        return (param_true, X_train, y_train, X_test, y_test) 
        
