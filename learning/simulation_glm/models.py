import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from scipy.optimize import NonlinearConstraint, LinearConstraint, minimize
from scipy.special import expit

# import dependencies
from dataset_generator import ExponentialFamilies
import warnings

def Sigmoid(x):
    return expit(x)

class ActiveLearningInfluenceMax(object):
    def __init__(self, args = None):
        if args is None:
            self.input_dim  = 20
            self.covtype    = "PPCA"
            self.imax_bsize = 10
            self.optx       = True
            self.finite     = True
        else:
            self.input_dim  = args.input_dim
            self.covtype    = args.covtype
            self.imax_bsize = args.imax_bsize
            self.optx       = args.optx
            self.finite     = args.finite
            
      
    def getLOOBiasEst(self, X, y, upper_bound=None):
        n = len(X)
        out = 0
        mask = np.zeros(n, dtype = bool)
        
        if upper_bound is None:
            upper_bound = n

        if n <= upper_bound:
            if n == 1:
                out = torch.ones_like(X.T)
            else: 
                for j in range(n):
                    mask[j] = True
                    out += self.getCurrentEstParam(X[~mask], y[~mask])
                    mask[j] = False
            out /= n
        else:
            idx_jacknife = np.random.choice(np.arange(n), size=upper_bound, replace=False)
            for j in range(upper_bound):
                mask[idx_jacknife[j]] = True
                out += self.getCurrentEstParam(X[~mask], y[~mask])
                mask[idx_jacknife[j]] = False
            out /= upper_bound
        return out


    def getCurrentEstParam(
        self, 
        X, 
        y, 
        Xtest = None, 
        ytest = None, 
        evaluation = False
    ):
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = LogisticRegression().fit(X[:, 1:].numpy(), y.squeeze(1).numpy())
            if evaluation:
                predicted = torch.from_numpy(clf.predict(Xtest[:, 1:].numpy()))
                acc = (predicted  == ytest.squeeze()).sum() / len(ytest)
                return torch.from_numpy(np.concatenate((clf.intercept_, clf.coef_.squeeze()))).unsqueeze(1), acc
            else:
                return torch.from_numpy(np.concatenate((clf.intercept_, clf.coef_.squeeze()))).unsqueeze(1)

    def acquireNext(
        self, 
        Xtest, 
        Xtrain_acquired, 
        B_curr, 
        B_true_est, 
        B_true = None, # to generate the next sample to acquire in continuous space case
        Xtrain = None, # to find the closest samples in the finite case if using optimization over the continuous space
        mask = None
    ):
        """
        args.finite: True/False
        args.optx:   True/False
        
        - args.finite = True 
          - args.optx = True:  we optimize over x with batch size args.imax_bsize, 
                               find the closest x's in the finite remaining pool
          - args.optx = False: we find the best in the finite remaining pool
          
          return: idx_new

        - args.finite = False  (continuous space)
          - args.optx = True:  we optimize over x with batch size args.imax_bsize
          
          return: new_x, new_y 

        """

        # Compute the gradient on the expected test loss
        ExpFCurr = ExponentialFamilies(
            X = Xtest, 
            param_true = B_true_est, 
            test = True, 
            expected = True
        ) 
        grad_ETestLoss = ExpFCurr.getGradLoss(param = B_curr)
        
        
        if self.optx: 
            # Explicit Hessian computation using the currently acquired training samples 
            eta_curr = torch.matmul(Xtrain_acquired, B_curr)
        
            h = torch.sigmoid(eta_curr) * (1 - torch.sigmoid(eta_curr))
            H = torch.matmul(Xtrain_acquired.T, Xtrain_acquired * h)  + torch.eye(self.input_dim + 1) * 1e-9
            
            grad_ETestLoss = grad_ETestLoss.numpy()
            H              = H.numpy()
            B_true_est     = B_true_est.numpy().squeeze(1)
            B_curr         = B_curr.numpy().squeeze(1)
            y = np.linalg.solve(H, grad_ETestLoss)

            index_ins = np.arange(0, self.input_dim*self.imax_bsize, self.input_dim)[:self.imax_bsize]
            index_out = np.arange(0, self.input_dim*(self.imax_bsize + 1), self.input_dim+1)[:self.imax_bsize]
            def opt_overx(x):
                x = np.insert(x, index_ins, np.ones((self.imax_bsize)))
                b  = np.dot(x, np.repeat(B_curr, self.imax_bsize))
                bT = np.dot(x, np.repeat(B_true_est, self.imax_bsize))
                I  = (Sigmoid(b) - Sigmoid(bT)) * np.dot(x, np.repeat(y, self.imax_bsize))
                return -I

            def opt_overx_grad(x):
                x = np.insert(x, index_ins, np.ones((self.imax_bsize)))
                B_curr_expand = np.repeat(B_curr, self.imax_bsize)
                B_true_est_expand = np.repeat(B_true_est, self.imax_bsize)
                y_expand = np.repeat(y, self.imax_bsize)
                b  = np.dot(x, B_curr_expand)
                bT = np.dot(x, B_true_est_expand)

                temp1 = (Sigmoid(b) *(1 - Sigmoid(b)) * B_curr_expand
                        - Sigmoid(bT) * (1 - Sigmoid(bT)) * B_true_est_expand)
                temp2 = (Sigmoid(b) - Sigmoid(bT)) * y_expand

                temp1 = temp1 * np.dot(x, y_expand)
                grad  = - temp1 - temp2
                grad = np.delete(grad, index_out)
                return grad
                
            def opt_overx_hess_p(x, p):
                x = np.insert(x, index_ins, np.ones((self.imax_bsize)))
                
                B_curr_expand = np.repeat(B_curr, self.imax_bsize)
                B_true_est_expand = np.repeat(B_true_est, self.imax_bsize)
                p = np.insert(p, index_ins, np.zeros((self.imax_bsize)))
                b  = np.dot(x, B_curr_expand)
                bT = np.dot(x, B_true_est_expand)

                y_expand = np.repeat(y, self.imax_bsize)
                temp1 = (Sigmoid(b) *(1 - Sigmoid(b)) * B_curr_expand
                        - Sigmoid(bT) * (1 - Sigmoid(bT)) * B_true_est_expand)
                temp1 = temp1 * np.dot(y_expand, p)

                temp3 = y_expand * np.dot(temp1, p)

                temp21 = Sigmoid(b) * ((1-Sigmoid(b))**2 
                                    - Sigmoid(b) * (1 - Sigmoid(b)))
                temp22 = Sigmoid(bT) * ((1-Sigmoid(bT))**2 
                                        - Sigmoid(bT) * (1 - Sigmoid(bT)))
                temp2  = (temp21 * B_curr_expand * np.dot(B_curr_expand, p) 
                        - temp22 * B_true_est_expand * np.dot(B_true_est_expand, p))
                temp2  *= np.dot(x, y_expand)

                hess_p = - temp1 - temp2 - temp3 
                hess_p = np.delete(hess_p, index_out)
                return hess_p

            con = lambda x: np.sum(x ** 2)
            if self.covtype == "high-corr":
                ub = 200 * self.imax_bsize
                x0b = 6
            elif self.covtype == "PPCA":
                ub = 510 * self.imax_bsize
                x0b = 8
            else: 
                raise ValueError(f'covtype should either be "high-corr" or "PPCA" at this stage. Received: {self.covtype}.')
            nonlinear_constraint = NonlinearConstraint(con, 0, ub)
            linear_constraint = LinearConstraint(np.zeros((self.imax_bsize * self.input_dim)), 0, 0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                x0 = np.random.uniform(-x0b, x0b, self.imax_bsize * self.input_dim)
                if len(Xtrain_acquired) > 500:
                    res = minimize(opt_overx, x0, method = 'trust-constr', 
                                    jac = opt_overx_grad, hessp = opt_overx_hess_p,
                                    constraints = [linear_constraint, nonlinear_constraint],
                                    options = {'xtol': 1e-4, 'gtol': 1e-4}) # options = {'maxiter': 500}
                else:
                    res = minimize(opt_overx, x0, method = 'trust-constr', 
                                    jac = opt_overx_grad, hessp = opt_overx_hess_p,
                                    constraints = [linear_constraint, nonlinear_constraint]) # options = {'maxiter': 500}
                
            new_x = torch.from_numpy(res.x.reshape(-1, self.input_dim))

            if self.finite:
                msediff = - 2 * Xtrain[~mask, 1:] @ new_x.T
                msediff_1 = torch.sum(Xtrain[~mask, 1:]**2, dim = 1)
                msediff_2 = torch.sum(new_x**2, dim = 1)
                msediff += (msediff_1.unsqueeze(-1) + msediff_2.unsqueeze(0))
                idx_new = torch.argmin(msediff, dim = 0)

                # new index in the whole pool 
                idx_new = np.arange(len(mask))[~mask][idx_new]
            else:
                g = torch.matmul(new_x, B_true[1:]) + B_true[0]
                mu = torch.sigmoid(g)
                new_y = torch.bernoulli(mu)
            
        else: # This is only for finite case
            if self.finite:
                # Compute the gradient on the expected training loss for all the remaining training data points
                ExpFCurr = ExponentialFamilies(
                    X = Xtrain[~mask], 
                    param_true = B_true_est, 
                    test = False, 
                    expected = True
                )
                
                grad_ETrainLoss = ExpFCurr.getGradLoss(param = B_curr)

                # Explicit Hessian computation using the currently acquired training samples 
                ExpFCurr = ExponentialFamilies(X = Xtrain[mask]) 
                eta_curr = ExpFCurr.getEta(B_curr)
                h = torch.sigmoid(eta_curr) / (1+torch.exp(eta_curr))
                H = torch.matmul(Xtrain[mask].T, Xtrain[mask] * h)  + torch.eye(self.input_dim + 1) * 1e-9
            
                # Compute the influence function value on the all the remaining datapoints and obtain the next acquire datapoint
                I = torch.matmul(grad_ETestLoss, torch.solve(grad_ETrainLoss.T, H)[0])
                # new index in the remaining pool
                idx_new = torch.topk(I, k = self.imax_bsize)[1]

                # new index in the whole pool 
                idx_new = np.arange(len(mask))[~mask][idx_new]
        
    
            else: 
                raise ValueError("When 'optx == False', we must have 'finite == True'. Received: optx{}, finite{}"
                                 .format(self.optx, self.finite))
        if self.finite: 
            return idx_new
        else: 
            return new_x, new_y

