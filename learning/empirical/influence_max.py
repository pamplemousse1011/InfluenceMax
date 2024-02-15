
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.functional as Fag
import torch.autograd as AG
from torch.utils.data import DataLoader
from torch import Tensor

from model_classifier import MakeModule
from data_generator import get_targets, get_inputs
from utils import gc_cuda

from tqdm import tqdm

import dataclasses
import typing


@dataclasses.dataclass
class AcquisitionBatch:
    indices: typing.List[int]
    scores: typing.List[float]

def fast_inverse_vhp(func, inputs, v, J, T, progress_bar=False):
    """
    func  : function to compute hessian
    inputs: (tuple) inputs to the function func
    v     : (tuple) the vector for which the inverse hessian vector product is computed
    J     : number of recursive computation so that approximation converges
    T     : number of independent runs whose results to be averaged  
    """
    num_layers = len(v)
    out = [torch.zeros_like(item) for item in v]
    for _ in range(T):
        # initialize the inverse HVP estimation Hhat_0^{-1}v = v
        ivhp = v
        for _ in tqdm(range(J),disable=not progress_bar):
            temp = Fag.vhp(
                func, 
                inputs, 
                ivhp,
                strict = True
            )[1]
            ivhp = tuple([v[i] + ivhp[i] - temp[i] for i in range(num_layers)])
        out = [out[i] + ivhp[i] for i in range(num_layers)]  
    out = [item/T for item in out]  
    return out  

def model_reconstruct(x, param):
    x = torch.flatten(x, start_dim=1)
    for i in range(len(param)//2 - 1):
        x = F.linear(x, weight=param[i*2], bias=param[i*2+1])
        x = F.relu(x)
        
    i_last = len(param)//2 - 1
    x = F.linear(x, weight=param[i_last*2], bias=param[i_last*2+1])
    return x

def transform_output(x, type="None"):
        if type == "None":
            return x
        if type == "softmax": 
            return F.softmax(x, -1)
        if type == "log_softmax":
            return F.log_softmax(x, -1)
        
        raise NotImplementedError(f"Unknown type {type}!")

def compute_pred(
        data, 
        dtype,
        num_classes: int,
        model : nn.Module = None,
        param_reconstruct = None, 
        output_type: str = "None",
        device = None,
    ):  

        if param_reconstruct:
            # here ```param``` is ```model.parameters()```
            # if isinstance(param_reconstruct[0], tuple):
            #     param_reconstruct = param_reconstruct[0]
            def model(x):
                return model_reconstruct(x=x, param=param_reconstruct)
        else:
            model.to(device)
        if isinstance(data, DataLoader): 
            N = len(data.dataset)

            # This stays on the CPU?
            out = torch.empty(
                (N, num_classes), 
                dtype=dtype, 
                device=device
            )
            
            for batch_idx, (batch_data, _) in enumerate(data):
                left  = batch_idx*data.batch_size
                right = min((batch_idx+1)*data.batch_size, N)
                
                batch_data = batch_data.to(device)
                # batch_size x num_classes
                output_B = model(batch_data)
                
                
                output_B = transform_output(x=output_B, type=output_type)
                if not param_reconstruct:
                    output_B = torch.mean(output_B, dim = 0)
                out[left:right] = output_B
        elif isinstance(data, list):
            data = data[0].to(device)
            out = model(data)
            
            out = transform_output(x=out, type=output_type)
            if not param_reconstruct:
                out = torch.mean(out, dim = 0)
            
        elif isinstance(data, Tensor):
            data = data.to(device)
            out = model(data)
 
            out = transform_output(x=out, type=output_type)
            if not param_reconstruct:
                out = torch.mean(out, dim = 0)
            
        else:
            raise ValueError(f"Unknown data type {type(data)}!")
        return out



class InfluenceMax():
    def __init__(self,
                 model: nn.Module,
                 train_eval_loader: DataLoader,
                 available_loader_Ey: DataLoader,
                 available_loader_hess: DataLoader,
                 test_loader: DataLoader,
                 num_classes: int,
                 args,
                 device):
        self.model             = model 
        self.num_classes       = num_classes
        self.args              = args
        self.device            = device
        self.dtype             = train_eval_loader.dataset[0][0].dtype
        self.num_acqd  = len(train_eval_loader.dataset)
        if self.num_acqd < 5000:
            self.data_acqd =list([get_inputs(train_eval_loader.dataset).to(self.device),
                                  get_targets(train_eval_loader.dataset).to(self.device)])
        else:
            self.data_acqd = train_eval_loader
       
        if len(test_loader) == 1:
            self.data_test =list([get_inputs(test_loader.dataset).to(self.device),
                                  get_targets(test_loader.dataset).to(self.device)])
        else:
            self.data_test = test_loader
        self.num_test  = len(test_loader.dataset)
        

        self.num_pool = len(available_loader_Ey.dataset)
        self.data_pool_Ey   = available_loader_Ey
        self.data_pool_hess = available_loader_hess
       
        
    
    
    def loss_acqd(self, *param):
        out_acqd = compute_pred(
            num_classes=self.num_classes,
            param_reconstruct=param, 
            data=self.data_acqd,   
            output_type="None",
            dtype=self.dtype,
            device=self.device
        )
        if isinstance(self.data_acqd, DataLoader):
            y_acqd = get_targets(self.data_acqd.dataset).to(self.device)
            return F.cross_entropy(input=out_acqd, target=y_acqd)
        if isinstance(self.data_acqd, list):
            return F.cross_entropy(input=out_acqd, target=self.data_acqd[1])
        
        raise NotImplementedError(f"Received data_acqd type {type(self.data_acqd)}!")
        
    def compute_GETaskLoss(self, model, data, Ey):
        logpi_task=compute_pred(
            data=data,
            dtype=self.dtype,
            num_classes=self.num_classes,
            model=model, 
            output_type="log_softmax",
            device=self.device
        ) # of size num_test x num_classes
        # For task loss, we average over all task data (as well as all C classes)
        ETaskLoss=-torch.mean(logpi_task*Ey) # scalar
        # Compute the gradient of expected task loss over the model parameters
    
        GETaskLoss = AG.grad(
            ETaskLoss, 
            model.parameters(), 
            retain_graph=True
        ) # self.model.nets.parameters, the same
        gc_cuda()
        return GETaskLoss
        
    def compute_GEPoolLoss(self, model, data, Ey):
        def loss_pool(*param):
            logpi_pool = compute_pred(
                data=data, 
                dtype=self.dtype,
                param_reconstruct=param, 
                output_type="log_softmax",
                device=self.device,
                num_classes=self.num_classes
            ) # of size num_available x num_classes
            # For train loss on all the remaining training data points, we only average over C classes
            EPoolLoss = -torch.mean(logpi_pool*Ey, dim=-1) # num_available x 1 
            return EPoolLoss
            
        GEPoolLoss = Fag.jacobian(
            loss_pool, 
            tuple(model.parameters())
        )
        gc_cuda()
        return GEPoolLoss

    def compute_influence(self, I_temp, model, data, Ey):
        if isinstance(data, DataLoader):
            num_data = len(data.dataset)
            I = torch.zeros(
                (num_data,), 
                device = self.device, 
                dtype  = self.dtype
            )

            for batch_idx, (batch_data, _) in enumerate(tqdm(data, disable=not self.args.progress_bar)):
                left  = batch_idx*data.batch_size
                right = min((batch_idx+1)*data.batch_size, num_data)
                batch_data = batch_data.to(self.device)
                GEPoolLoss = self.compute_GEPoolLoss(
                    model=model,
                    data=batch_data, 
                    Ey=Ey[left:right]
                )
                
                for i in range(len(I_temp)): 
                    I[left:right] += (GEPoolLoss[i]*I_temp[i]).view(right-left, -1).sum(dim=1)
        else:     
            num_data = len(data)
            GEPoolLoss = self.compute_GEPoolLoss(
                data=data, 
                Ey=Ey,
                model=model
            )
            for i in range(len(I_temp)):
                I += (GEPoolLoss[i]*I_temp[i]).view(num_data, -1).sum(dim=1)
        gc_cuda()
        return I
    
    def compute_scores(self):
        print(f"Step 1: Obtain the expected values of the POOL data and the TEST data...")
        with torch.no_grad():
            module_from_list = MakeModule(self.model.nets[1:])
            # We need all models to compute the expected values
            Ey_task = compute_pred(
                data = self.data_test, 
                dtype = self.dtype,
                num_classes = self.num_classes,
                model = module_from_list,
                output_type = "softmax",
                device = self.device
            )
            Ey_pool = compute_pred(
                data = self.data_pool_Ey, 
                dtype = self.dtype,
                num_classes = self.num_classes,
                model = module_from_list,
                output_type = "softmax",
                device = self.device
            )
            module_from_list = None
        
        print(f"Step 2: Compute the gradient of expected loss for the TEST data...")
        # We only need the first models to compute the expected values
        GETaskLoss = self.compute_GETaskLoss(
            model=self.model.nets[0], 
            data=self.data_test, 
            Ey=Ey_task
        ) 

        print(f"Step 3: Compute the Hessian inverse vector product...")    
        if self.num_acqd <= 5000:
            I_ivhp = fast_inverse_vhp(
                func=self.loss_acqd, 
                inputs=tuple(self.model.nets[0].parameters()), 
                v=GETaskLoss, 
                J=self.args.J, 
                T=self.args.T,
                progress_bar=self.args.progress_bar
            )
            
            print(f"Step 4: Compute the gradient of expected loss for the POOL data and the influence scores...")
            I = self.compute_influence(
                I_temp=I_ivhp, 
                model=self.model.nets[0], 
                data=self.data_pool_hess, 
                Ey=Ey_pool
            )
        else:
            raise ValueError("Number of acquired data is larger than 5000!!!!")  

        gc_cuda() 
        return I 

    def compute_infmax_batch(self): 
        I = self.compute_scores()
        global_acquision_scores, global_acquisition_bag = torch.topk(
            I.squeeze(-1), k = self.args.available_sample_k
        )
        # global_acquisition_bag = [indice.item() for indice in global_acquisition_bag]
        # global_acquision_scores = [score.item() for score in global_acquision_scores]
        return AcquisitionBatch(
            global_acquisition_bag.cpu().numpy(), 
            global_acquision_scores.cpu().numpy()
        )


    
