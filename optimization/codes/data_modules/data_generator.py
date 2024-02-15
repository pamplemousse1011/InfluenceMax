from __future__ import division 
import time 
from abc import ABC, abstractmethod
from typing import List, Callable, Union

import numpy as np
import numpy.random as npr
 
import torch 
 
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint


from utils import generate_seed_according_to_time, gc_cuda
from networks.pretrained_resnet import PretrainedFeatEmbModule
from data_modules.image_model import *
from data_modules.model_utils import *
from data_modules.image_dataset import *
from data_modules.data_utils import lowrank_perb 

MAXEPOCHS = {
    'cnn'     : 50,
    'wrn'     : 50,
    'resnet'  : 50,
    'preactrn': 50,
    'lenet'   : 50,
    'vgg'     : 50,
}

def generate_samples(
        n:int, 
        search_domain:np.ndarray, 
        method='random', 
        low_dim=0, 
        seed=generate_seed_according_to_time(1)[0]
    ) -> np.ndarray: 
    """
    Inputs
        n: Number of samples to generate.
        search_domain: (ndim, 2)
        method: 'grid' or 'random' or 'lhs'
    Returns:
        (n**ndim, ndim) if method == 'grid' else (n, ndim)

    """

    d = search_domain.shape[0]
    rng = npr.default_rng(seed)

    if method == 'lowrank':
        assert low_dim > 0, f"using 'lowrank' in generate_samples, low_dim has to be larger than 0! Received low_dim={low_dim}."
        samples = lowrank_perb(n, low_dim, d, search_domain, rng=rng)
    if method == 'grid':
        ls = [np.linspace(lo, hi, n, endpoint=True) for (lo, hi) in search_domain]
        mesh_ls = np.meshgrid(*ls)
        all_mesh = [np.reshape(x, [-1]) for x in mesh_ls]
        samples = np.stack(all_mesh, axis=1) 
    elif method == 'random':
        samples = np.zeros((n, d))
        samples = rng.uniform(0, 1, size=(n, d)) 
        samples = samples*(search_domain[:,1]-search_domain[:,0])+search_domain[:,0]
    elif method == 'lhs':
        """Latin hypercube sampling (LHS).
        It generates n points in [0,1)^d. Each univariate marginal distribution is 
        stratified, placing exactly one point in [j/n, (j+1)/n] for j = 0,1,...,n-1.

        When LHS is used for integrating a function f over n, 
        LHS is extremely effective on integrands that are nearly additive. 
        With a LHS of n points, the variance of the integral is always lower than 
        plain MC on n-1 points. There is a central limit theorem for LHS 
        on the mean and variance of the integral, but not necessarily for 
        optimized LHS due to the randomization.
        """
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=d, seed=seed)
        samples= qmc.scale(sampler.random(n), search_domain[:,0], search_domain[:,1])
    else: 
        raise ValueError(f"Received method={method}; can only be 'grid' or 'random' or 'lhs'.")
    
    return samples.astype(search_domain.dtype) 

class OptimizationDataGenerator(ABC):
    def __init__(
        self,  
        use_double           : bool=False,
        seed                 : int=101, 
        **kwargs
    ):  
        if use_double:
            self.npdtype = np.float64
            self.dtype = torch.float64
        else:
            self.npdtype = np.float32
            self.dtype = torch.float32
        
        ## Set the seed
        # create the RNG that you want to pass around
        rng = npr.default_rng(101)
        # get the SeedSequence of the passed RNG
        ss = rng.bit_generator._seed_seq
        # create 10 initial independent states for 10 total experiments 
        self.child_states = ss.spawn(5000) 
        # seed to generate samples
        self.seed   = seed
        # optimum
        self._f_opt = 0
    
    @abstractmethod
    def gen_samples(self, n) -> np.ndarray:
        pass

    @abstractmethod
    def init_attributes(self) -> None: 
        pass 

    @abstractmethod
    def post_init_steps(self) -> None:
        pass 
    
    @abstractmethod
    def evaluate(self, x:np.ndarray, seed:int=None):
        pass
        
class ImageDataGenerator(OptimizationDataGenerator):
    """
    Arguments: 
    ================
    noise_level: float positive 
        sample var = noise_level * var(y) 
    high_dim: int 
        Number of total inputs. 
        See _high_dim and _active_dim in Attributes  
    low_dim: int
        Number of variables to generate 
        the lowrank high-dimensional inputs
    seed: int 
        To generate random state
    sample_var: float (None by default) 
        If sample_var = -1, noise_level * var(y),
        where y of size 10000 are randomly sampled.  

    Attributes:
    ================
    _high_dim: int
        Number of total inputs. 
    _active_dim: int 
        Number of active variables in Branin function
    _sample_var: float
        Noise variance  
    """
    def __init__(
        self, 
        use_double           : bool=False,
        seed                 : int=101,
        net_type             : str='cnn',
        n_small_data         : int=0,
        data_dir             : str='/home/data/',
        log_dir              : str='/home/log/',
        log_interval         : int=10, 
        use_validation       : bool=False, 
        num_workers          : int=5,
        pin_memory           : bool=False,
        n_devices            : int=1,
        progress_bar         : bool=False, 
        **kwargs
    ):   
        super().__init__(use_double, seed, **kwargs)
        self.search_domain = np.array([[0.,1.],[-4,-1.],[-4.,-1.],[0.,1.]]).astype(self.npdtype)
        ## Image model  
        self.n_small_data = n_small_data
        self.data_dir = data_dir
        self.log_dir = log_dir
        self.use_validation = use_validation
        self.net_type = net_type
        self.device = "gpu" if torch.cuda.is_available() else "cpu"
        self.max_epochs = MAXEPOCHS[net_type]
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        ## Creating the trainer...
        self.kwargs_trainer = {
            'max_epochs': self.max_epochs, 
            'min_epochs': min(50, self.max_epochs),
            'accelerator': self.device,
            'enable_model_summary': True,
            'check_val_every_n_epoch': log_interval if use_validation else 1,
            'default_root_dir': log_dir,
            'enable_checkpointing': True,
            'enable_progress_bar': progress_bar,
            'logger': True,
            'deterministic': False,
            'devices': n_devices,
        }

    def init_attributes(self):
        self.num_classes = None
        self.rgb = None
        self.dm = None
        # base model 
        self.base_model = None

    def post_init_steps(self):
        if self.dm is None:
            raise AttributeError("Attribute 'dm' must be set in the child class.")
        if self.num_classes is None:
            raise AttributeError("Attribute 'num_classes' must be set in the child class.")
        if self.rgb is None:
            raise AttributeError("Attribute 'rgb' must be set in the child class.")
        
    def gen_samples(self, n, seed=None) -> np.ndarray:
        if seed is None: 
            seed = self.child_states[self.seed]
        return generate_samples(n, self.search_domain, seed=seed)
    
    def load_model(self, x:np.ndarray, save:bool=False, log_filepath:str=None, minfo_filepath:str=""):
        """
        x: (d,)
        """
        if os.path.isfile(minfo_filepath):
            image_model = ImageModelModule.load_from_checkpoint(minfo_filepath, map_location="cpu") 
            print("Found and loaded model from path {}".format(minfo_filepath))
        else:
            momentum, lr, wd, dropout_rate = x 
            lr = pow(10, lr)
            wd = pow(10, wd)  

            dropout_rate = np.clip(dropout_rate, None, 0.99)

            image_model = ImageModelModule(
                self.net_type,
                self.num_classes,
                self.rgb,
                dropout_rate,
                lr, 
                wd,
                momentum,
                self.max_epochs, 
                save_logpath=log_filepath
            ) 
            
            if save:
                assert minfo_filepath is not None, "When save=True, minfo_filepath cannot be None!"
                print("Could not find model path {}. Creating model and save the best...".format(minfo_filepath))
                # Save the top 1 checkpoint that give the smallest values of training or validation loss
                monitor_value = 'val_loss' if self.use_validation else 'loss'
                checkpoint_callback = [
                    ModelCheckpoint(
                            monitor=monitor_value,
                            dirpath=self.log_dir,
                            filename=os.path.splitext(minfo_filepath)[0], # '-'.join([minfo_filepath.split('.')[0],'best']),
                            auto_insert_metric_name=False,
                            # save_last=True,
                            save_top_k=1,
                            mode='min',
                        )
                ]
            else: 
                print("Creating model ... (will not save)".format(minfo_filepath))
                checkpoint_callback = None

            # Fitting the model...
            t0 = time.time()
            trainer = pl.Trainer(**self.kwargs_trainer, callbacks=checkpoint_callback)
            trainer.fit(model=image_model, datamodule=self.dm) 
            t1 = time.time()-t0
            print("Training takes {:.3f}s.".format(t1)) 
            
            # Saving the model
            # if save:
            #     # Modify the last checkoint saved name
            #     last_checkpoint_path = os.path.join(trainer.checkpoint_callback.dirpath, 'last.ckpt')
                
            #     if os.path.exists(last_checkpoint_path):
            #         os.rename(last_checkpoint_path, minfo_filepath)
            #     else:
            #         raise ValueError(f"The 'last' checkpoint cannot be found in {trainer.checkpoint_callback.dirpath}!")
            #     print("Saved model in {}...".format(minfo_filepath))
            #     # if model_choice == "best":
            #     #     best_checkpoint_path = os.path.join(trainer.checkpoint_callback.dirpath, model_best_filename)
            #     #     if os.path.exists(best_checkpoint_path):
            #     #         image_model.load_from_checkpoint(best_checkpoint_path)
            #     #     else:
            #     #         raise ValueError(f"The '{model_choice}' checkpoint cannot be found in {trainer.checkpoint_callback.dirpath}!")
        return image_model.eval()
    
    def get_base_x_embedding(self, pretrained_featemb_model_path:str="", return_logits:bool=False) -> nn.Module:
        if os.path.isfile(pretrained_featemb_model_path):  
            print(f"Generating base x embedding from pretrained feature embedding model from {pretrained_featemb_model_path} (use_logits={str(return_logits)[0]})...")
            featemb_model = PretrainedFeatEmbModule(
                filepath=pretrained_featemb_model_path,
                num_classes=self.num_classes,
                num_input_channels=3 if self.rgb else 1,
                return_logits=return_logits)
                
            if not os.path.isfile(os.path.join(self.log_dir, f'precomputed_featemb-train.pt')):
                trainer = pl.Trainer(**self.kwargs_trainer)
                all_outputs = trainer.predict(model=featemb_model, datamodule=self.dm)
                
                ## SAVE FEATURE EMBEDDING FOR BOTH TRAIN AND TEST 
                for dd in range(len(all_outputs)):
                    output_filename = 'precomputed_featemb-{}.pt'.format(self.dm._data[dd])
                    output_filepath = os.path.join(self.log_dir, output_filename) 
                    torch.save(torch.concatenate(all_outputs[dd]), output_filepath)  
            return lambda x:x, featemb_model.n_features
        else:
            print(f"Cannot load pretrained feature embedding model from {pretrained_featemb_model_path}.")
            print("Initiate a feature embedding model from the target image model...")
            featemb_model = ImageModelModule(
                self.net_type,
                self.num_classes,
                self.rgb,
                0,
                0.001, 
                0.001,
                0.9,
                1, 
                save_logpath=None
            ).featemb
            return featemb_model, featemb_model.n_features
        # minfo_filepath = os.path.join(self.log_dir, 'i{:04d}_model-best.ckpt'.format(use_i)) 
        # image_model = ImageModelModule.load_from_checkpoint(minfo_filepath, map_location="cpu").featemb  
        # ## LOAD TRAINER
        # trainer = pl.Trainer(**self.kwargs_trainer)

        # ## MAKE PREDICTIONS    
        # _ = trainer.test(model=image_model, datamodule=self.dm)
        # for dd in range(len(self.dm.get_embedding_data)): 
        #     output_filename = 'i{:04d}_emb-{}.pt'.format(use_i, self.dm.get_embedding_data[dd])
        #     output_filepath = os.path.join(self.log_dir, output_filename)
        #     torch.save(image_model.base_x_embedding[dd], output_filepath)
        # del image_model
        # return None
    
    def evaluate_all_and_save_individual(self, x:np.ndarray, i:int) -> List[np.ndarray]:
        """
        x: (d,) or (1,d)
        """
        x = x.reshape(-1)
        log_filepath   = os.path.join(self.log_dir, 'i{:04d}_log.pt'.format(i))
        minfo_filepath = os.path.join(self.log_dir, 'i{:04d}_model-best.ckpt'.format(i)) 
        image_model = self.load_model(x, save=False, log_filepath=log_filepath, minfo_filepath=minfo_filepath)
        
        ## LOAD TRAINER
        trainer = pl.Trainer(**self.kwargs_trainer)
        
        ## MAKE PREDICTIONS    
        all_outputs = trainer.predict(model=image_model, datamodule=self.dm)
        
        del image_model, trainer
        gc_cuda()

        ## SAVE EACH LOSS FOR BOTH TRAIN AND TEST, AND RETURN AVERAGE LOSS
        ret = []
        for dd in range(len(all_outputs)):
            output = torch.concatenate(all_outputs[dd]) 
            # in case there is nan values
            output[torch.isnan(output)]=9.

            ret.append(output.mean().numpy())

            output_filename = 'i{:04d}_output-{}.pt'.format(i, self.dm._data[dd])
            output_filepath = os.path.join(self.log_dir, output_filename)
            output_dict = {"y": output, "ymean": output.mean(), "yvar": output.var()}
            torch.save(output_dict, output_filepath)
            
        del all_outputs
        gc_cuda()
        return ret
        
    def evaluate(self, x:np.ndarray) -> np.ndarray:
        """
        x: (d,) or (1,d)
        """
        image_model = self.load_model(x.reshape(-1))
 
        ## LOAD TRAINER
        trainer = pl.Trainer(**self.kwargs_trainer)

        ## MAKE PREDICTIONS    
        all_outputs = trainer.predict(model=image_model, datamodule=self.dm)

        del image_model, trainer
        gc_cuda()

        ret = []
        for dd in range(len(all_outputs)):
            output = torch.concatenate(all_outputs[dd]) 
            # in case there is nan values
            output[torch.isnan(output)]=9.
            
            ret.append(output.mean().numpy())

        del all_outputs
        gc_cuda()
        return ret 
         
class CIFAR10(ImageDataGenerator):
    def __init__(
        self,  
        use_double           : bool=False,
        seed                 : int=101,
        net_type             : str='cnn',
        n_small_data         : int=0,
        data_dir             : str='/home/data/',
        log_dir              : str='/home/log/',
        log_interval         : int=10,
        use_validation       : bool=False, 
        num_workers          : int=5,
        pin_memory           : bool=False,
        n_devices            : int=1,
        progress_bar         : bool=False,
        shift                : bool=False,
        targets_to_shrink    : list=[1,2,7],
        shrink_to_proportion : float=0.1,
        in_distribution      : bool=False,
        **kwargs
    ):   
        super().__init__(use_double, seed, net_type, n_small_data,
                         data_dir, log_dir, log_interval, use_validation, 
                         num_workers, pin_memory, n_devices, progress_bar, **kwargs)
        self.shift= shift
        self.targets_to_shrink = targets_to_shrink
        self.shrink_to_proportion = shrink_to_proportion
        self.in_distribution = in_distribution
        
        self.init_attributes()
        self.post_init_steps()

    def init_attributes(self):
        ## Creating the data module...
        self.dm = CIFAR10DataModule(
            batch_size=128, 
            n_small_data=self.n_small_data,
            dtype=self.dtype,
            data_dir=self.data_dir, 
            seed=self.seed,
            use_validation=self.use_validation, 
            ntrain=None,
            shift=self.shift, 
            targets_to_shift=self.targets_to_shrink,
            shrink_to_proportion=self.shrink_to_proportion,
            in_distribution=self.in_distribution,
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory
        )
        self.num_classes = 10
        self.rgb = True
         
class MNISTFRS(ImageDataGenerator):
    def __init__(
        self,  
        use_double           : bool=False,
        seed                 : int=101,
        net_type             : str='cnn',
        n_small_data         : int=0,
        data_dir             : str='/home/data/',
        log_dir              : str='/home/log/',
        log_interval         : int=10,
        use_validation       : bool=False, 
        num_workers          : int=5,
        pin_memory           : bool=False,
        n_devices            : int=1,
        progress_bar         : bool=False,
        **kwargs
    ):
        super().__init__(use_double, seed, net_type, n_small_data,
                         data_dir, log_dir, log_interval, use_validation, 
                         num_workers, pin_memory, n_devices, progress_bar,
                         **kwargs)
        self.init_attributes()
        self.post_init_steps()

    def init_attributes(self):
        self.rgb = False
        self.num_classes = 10
        ## Creating the data module...
        self.dm =  MNISTFRSDataModule(
            batch_size=128, 
            n_small_data=self.n_small_data,
            dtype=self.dtype,
            data_dir=self.data_dir, 
            seed=self.seed,
            use_validation=self.use_validation, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory
        )

class SVHNFRM(ImageDataGenerator):
    def __init__(
        self,  
        use_double           : bool=False,
        seed                 : int=101,
        net_type             : str='cnn',
        n_small_data         : int=0,
        data_dir             : str='/home/data/',
        log_dir              : str='/home/log/',
        log_interval         : int=10,
        use_validation       : bool=False, 
        num_workers          : int=5,
        pin_memory           : bool=False,
        n_devices            : int=1,
        progress_bar         : bool=False,
        **kwargs
    ):
        super().__init__(use_double, seed, net_type, n_small_data,
                         data_dir, log_dir, log_interval, use_validation, 
                         num_workers, pin_memory, n_devices, progress_bar,
                         **kwargs)
        self.init_attributes()
        self.post_init_steps()

    def init_attributes(self):
        self.rgb = False
        self.num_classes = 10
        ## Creating the data module...
        self.dm =  SVHNFRMDataModule(
            batch_size=128, 
            n_small_data=self.n_small_data,
            dtype=self.dtype,
            data_dir=self.data_dir, 
            seed=self.seed,
            use_validation=self.use_validation, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory
        )

class NoisyFunctionGenerator(OptimizationDataGenerator):
    def __init__( 
        self,  
        use_double: bool=False,
        seed: int=101,  
        active_dim: int=2,
        low_dim: int=2, 
        high_dim: int=0,
        sample_var: float=-1.0,
        noise_level: float=0.0, 
        **kwargs
    ): 
        """Noisy function generator 
        
        Arguments: 
        ================
        noise_level: float positive 
            sample var = noise_level * var(y) 
        high_dim: int 
            Number of total inputs. 
            See high_dim and active_dim in Attributes  
        low_dim: int
            Number of variables to generate 
            the lowrank high-dimensional inputs
        seed: int 
            To generate random state
        sample_var: float (None by default) 
            If sample_var = -1, noise_level * var(y),
            where y of size 10000 are randomly sampled.  

        Attributes:
        ================
        high_dim: int
            Number of total inputs. 
        active_dim: int 
            Number of active variables in Branin function
        sample_var: float
            Noise variance  
        """
        super().__init__(use_double, seed, **kwargs)

        self.active_dim = active_dim
        self.dim = max(high_dim, self.active_dim)
        self.low_dim = low_dim 
        
        # Randomization
        self.rng = npr.default_rng(self.child_states[self.seed])

        # Noise variance
        self.sample_var  = sample_var
        self.noise_level = noise_level
        
        # Initialize search domain
        self.search_domain = None 
        # Optimum and optima
        self._f_opt = -1.
        self._optima = np.zeros((1, self.dim)).tolist() 

    def post_init_steps(self) -> None:
        if self.search_domain is None:
            raise ValueError("search_domain has not been updated in the subclass!")
        
        if self.sample_var < 0:
            self.sample_var = self.compute_var(10000, self.noise_level)

    def compute_var(self, n_var, noise_multiple):
        """Compute the variance for the noise
        Arguments
        =============
        n_var: int 
            number of samples to precompute the var(y) 
            in noise-free setting

        noise_multiple: float positive
            sample var = noise_multiple * var(y)
        """
        x = self.gen_samples(n_var, evaluate=False)
        y = self.evaluate_true(x)
        return np.var(y) * noise_multiple

    def gen_samples(self, n_sample, evaluate=False): 
        if self.dim > self.active_dim:
            samples_x = lowrank_perb(
                n_sample, 
                self.low_dim, 
                self.dim, 
                self.search_domain,
                rng=self.rng
            )
        else:
            samples_x = self.rng.uniform(0, 1, size=(n_sample, self.active_dim)).astype(self.dtype)  
            samples_x = samples_x*(self.search_domain[:,1]-self.search_domain[:,0])+self.search_domain[:,0]
        if evaluate:
            samples_y = self.evaluate(samples_x)
            return samples_x, samples_y
        else:
            return samples_x
        
    @abstractmethod
    def evaluate_true(self, x) -> np.ndarray: 
        pass 

    def evaluate(self, x:np.ndarray):  
        out = self.evaluate_true(x)
        out += self.rng.normal(0, np.sqrt(self.sample_var), size=out.shape).astype(self.dtype)   
        return out 

class Branin(NoisyFunctionGenerator):
    def __init__( 
        self,  
        use_double: bool=False,
        seed: int=101,   
        active_dim: int=2,
        low_dim: int=2, 
        high_dim: int=0,
        sample_var: float=-1.0,
        noise_level: float=0.0, 
        **kwargs
    ): 
        super().__init__(use_double, seed, 
                         active_dim, low_dim, high_dim, 
                         sample_var, noise_level, 
                         **kwargs)
        # Update search domain
        self._search_domain = np.repeat(
            [[0., 1.]], self.dim, axis=0
        ).astype(self.dtype) 

        # Update optimum and optima
        self._f_opt = -1.0473938910927865
        self._optima = [[(-np.pi+5)/15, 12.275/15], # (0.1238938230940138, 0.8183333333333334)
                       [(np.pi+5)/15,2.275/15], # [0.5427728435726529, 0.15166666666666667]
                       [(9.42478+5)/15,2.475/15]] # [0.9616520000000001, 0.165]
        
        # Update sample_var for noise generation
        self.post_init_steps()

    def evaluate_true(self, x:np.ndarray): 
        """
        Only the first _active_dim inputs matter 
        """
        a = 1 / 51.95 
        b = 5.1 / (4 * (np.pi ** 2.0))
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8 * np.pi)
        q = 44.81 
        xtemp = 15 * x 
        xtemp[...,0] -= 5
        term1 = (xtemp[...,1]-b*(xtemp[...,0]**2.0)+c*xtemp[...,0]-r)**2.0
        term2 = s*(1-t)*np.cos(xtemp[...,0])-q
        y = (term1+term2)*a
        return y
    
class Dropwave(NoisyFunctionGenerator):
    def __init__( 
        self,  
        use_double: bool=False,
        seed: int=101,   
        active_dim: int=2,
        low_dim: int=2, 
        high_dim: int=0,
        sample_var: float=-1.0,
        noise_level: float=0.0, 
        **kwargs
    ):  
        super().__init__(use_double, seed, 
                         active_dim, low_dim, high_dim, 
                         sample_var, noise_level, 
                         **kwargs)
        
        # Update search domain
        self.search_domain = np.repeat(
            [[-5.12, 5.12]], self.dim, axis=0
        ).astype(self.dtype) 
        
        # Update optimum and optima
        self._f_opt = -1.
        self._optima = np.zeros((self.dim,)).tolist() 
        
        # Update sample_var for noise generation
        self.post_init_steps()
    
    def evaluate_true(self, x:np.ndarray): 
        """
        Only the first _active_dim inputs matter 
        """ 
	
        frac1 = 1 + np.cos(12*np.sqrt(x[...,0]**2+x[...,1]**2))
        frac2 = 0.5*(x[...,0]**2+x[...,1]**2) + 2
	
        y = -frac1/frac2
        return y

class GoldSteinPrice(NoisyFunctionGenerator):
    """GOLDSTEIN-PRICE FUNCTION
    The logarithmic form of the Goldstein-Price function, on [0, 1]^2
    used in Picheny, V., Wagner, T., & Ginsbourger, D. (2012). 
    (A benchmark of kriging-based infill criteria for noisy optimization),
    which has a mean of zero and a variance of one
    """
    def __init__( 
        self,  
        use_double: bool=False,
        seed: int=101,   
        active_dim: int=2,
        low_dim: int=2, 
        high_dim: int=0,
        sample_var: float=-1.0,
        noise_level: float=0.0, 
        scaled: bool=True,
        **kwargs
    ): 
        super().__init__(use_double, seed, 
                         active_dim, low_dim, high_dim, 
                         sample_var, noise_level, 
                         **kwargs)
        
        # Update search domain, optimum and optima
        self.scaled = scaled
        if self.scaled:  
            self.search_domain = np.repeat(
                [[0., 1.]], self.dim, axis=0
            ).astype(self.dtype) 
            self._f_opt = -3.129125550610585
            self._optima = [[0.50, 0.25]]  
        else:
            self.search_domain = np.repeat(
                [[-2., 2.]], self.dim, axis=0
            ).astype(self.dtype) 
            self._f_opt = 3.
            self._optima = [[0, -1.]] 
        
        # Update sample_var for noise generation
        self.post_init_steps()

    def evaluate_true(self, x:np.ndarray):
        """
        Only the first _active_dim inputs matter 
        """
        if self.scaled:
            x1bar = 4*x[...,0] - 2
            x2bar = 4*x[...,1] - 2

            fact1a = (x1bar + x2bar + 1)**2
            fact1b = 19 - 14*x1bar + 3*x1bar**2 - 14*x2bar + 6*x1bar*x2bar + 3*x2bar**2
            fact1 = 1 + fact1a*fact1b
            
            fact2a = (2*x1bar - 3*x2bar)**2
            fact2b = 18 - 32*x1bar + 12*x1bar**2 + 48*x2bar - 36*x1bar*x2bar + 27*x2bar**2
            fact2 = 30 + fact2a*fact2b
            
            prod = fact1*fact2
            
            y = (np.log(prod) - 8.693) / 2.427
        else:    
            fact1a = (x[...,0] + x[...,1] + 1)**2
            fact1b = (19 - 14*x[...,0] + 3*x[...,0]**2 
                    - 14*x[...,1] + 6*x[...,0]*x[...,1] 
                    + 3*x[...,1]**2)
            fact1 = 1 + fact1a*fact1b
                
            fact2a = (2*x[...,0] - 3*x[...,1])**2
            fact2b = (18 - 32*x[...,0] + 12*x[...,0]**2 
                    + 48*x[...,1] - 36*x[...,0]*x[...,1] + 27*x[...,1]**2)
            fact2 = 30 + fact2a*fact2b
            y = fact1*fact2
        return y 

class Ackley(NoisyFunctionGenerator):
    def __init__(
        self,  
        use_double: bool=False,
        seed: int=101,   
        active_dim: int=5,
        low_dim: int=5, 
        high_dim: int=0,
        sample_var: float=-1.0,
        noise_level: float=0.0,  
        **kwargs
    ): 
        super().__init__(use_double, seed, 
                         active_dim, low_dim, high_dim, 
                         sample_var, noise_level, 
                         **kwargs)
        # Update search domain
        self.search_domain = np.repeat(
            [[-5, 5]], self.dim, axis=0
        ).astype(self.dtype) 
        
        # Update optimum and optima
        self._f_opt = 0.0
        self._optima = np.zeros((1, self.dim)).tolist()

        # Update sample_var for noise generation
        self.post_init_steps()
        
    def evaluate_true(self, x:np.ndarray):
        """
        Only the first _active_dim inputs matter 
        """
        a = 20
        b = 0.2
        c = 2*np.pi
        out = -a*np.exp(-b*np.sqrt(np.mean(x[...,:self.active_dim]**2, axis=-1)))
        out += -np.exp(np.mean(np.cos(c*x[...,:self.active_dim]), axis=-1))
        out += a + np.exp(1) 
        return out 

class Rastr(NoisyFunctionGenerator):
    def __init__(
        self,  
        use_double: bool=False,
        seed: int=101,   
        active_dim: int=6,
        low_dim: int=6, 
        high_dim: int=0,
        sample_var: float=-1.0,
        noise_level: float=0.0,  
        **kwargs
    ): 
        super().__init__(use_double, seed, 
                         active_dim, low_dim, high_dim, 
                         sample_var, noise_level, 
                         **kwargs)
        
        # Update search_domain
        self._search_domain = np.repeat(
            [[-5.12, 5.12]], self.dim, axis=0
        ).astype(self.dtype) 

        # Update optimum and optima
        self._f_opt = 0. 
        self._optima = np.zeros((1, self.dim)).tolist()

        # Update sample_var for noise generation
        self.post_init_steps()

    def evaluate_true(self, x:np.ndarray): 
        """
        Only the first _active_dim inputs matter 
        """ 
        temp = np.sum(x[...,:self.active_dim]**2 
                      - 10*np.cos(2*np.pi*x[...,:self.active_dim]),
                      axis=-1)
	
        y = 10.*self.active_dim + temp
        return y 
    
class Hartmann6(NoisyFunctionGenerator):
    """
    The rescaled form of the 6-dimensional Hartmann function
    used in Picheny, V., Wagner, T., & Ginsbourger, D. (2012). 
    (A benchmark of kriging-based infill criteria for noisy optimization),
    which has a mean of zero and a variance of one.
    """
    def __init__(
        self,  
        use_double: bool=False,
        seed: int=101,   
        active_dim: int=6,
        low_dim: int=6, 
        high_dim: int=0,
        sample_var: float=-1.0,
        noise_level: float=0.0,  
        **kwargs
    ): 
        super().__init__(use_double, seed, 
                         active_dim, low_dim, high_dim, 
                         sample_var, noise_level, 
                         **kwargs)
        
        # Update search_domain
        self._search_domain = np.repeat(
            [[0., 1.]], self._dim, axis=0
        ).astype(self.dtype)

        # Update optimum and optima
        self._f_opt = -3.322368011391339
        self._optima = [[0.20169,0.150011,0.476874,0.275332,0.311652,0.6573]] 

        # Update sample_var for noise generation
        self.post_init_steps()

    def evaluate_true(self, x:np.ndarray):
        """
        Only the first _active_dim inputs matter 
        """
        alpha = np.array([1.00, 1.20, 3.00, 3.20],
                         dtype=self.dtype)
        A = np.array([[10.00,  3.00, 17.00,  3.50,  1.70,  8.00], 
                      [ 0.05, 10.00, 17.00,  0.10,  8.00, 14.00], 
                      [ 3.00,  3.50,  1.70, 10.00, 17.00,  8.00],
                      [17.00,  8.00,  0.05, 10.00,  0.10, 14.00]],
                      dtype=self.dtype)
        P = 1.0e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886], 
                               [2329, 4135, 8307, 3736, 1004, 9991],
                               [2348, 1451, 3522, 2883, 3047, 6650], 
                               [4047, 8828, 8732, 5743, 1091, 381]],
                               dtype=self.dtype)
        results = 0.0
        for i in range(4):
            inner_value = 0.0
            for j in range(4):
                inner_value += A[i, j] * pow(x[...,j] - P[i, j], 2.0)
            results += alpha[i] * np.exp(-inner_value)
        results = -(2.58 + results) / 1.94
        return results 
   