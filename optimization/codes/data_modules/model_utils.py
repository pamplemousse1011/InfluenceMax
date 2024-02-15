import torch   
import torch.nn as nn
from torch.utils.data import DataLoader

from networks.wide_resnet import *
from networks.preact_resnet import *
from networks.cnn import * 
from networks.resnet import *
from networks.lenet import *
from networks.vgg import *

IMGEMB_FACTORY = {
    "cnn": CNNEmbedding,
    "wrn": Wide_ResNetEmbedding,
    "lenet": LeNetEmbedding,
    "vgg": VGGEmbedding,
    "resnet": ResNetEmbedding,
    "preactrn": PreActResNetEmbedding
}

def get_metrics(net:nn.Module, dataloader: DataLoader, device:torch.device, individual:bool=False):
    """
    Return individual loss and label difference for each data in dataloader.
    Dataloader should not be shuffled.
    """

    net.to(device)
    net.eval()
     
    loss_list = []
    diff_list = []
    loss    = 0
    loss_sq = 0
    total   = 0
    correct = 0
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss_batch = nn.functional.cross_entropy(
                outputs, 
                targets, 
                reduction="none" if individual else "sum") 
            if individual:
                loss_list.append(loss_batch)
                _, pred_batch = torch.max(outputs, 1)  
                diff_list.append(pred_batch.eq(targets))
            else:
                loss    += loss_batch.item() 
                loss_sq += (loss_batch**2).item() 
                total   += targets.size(0)
                correct += pred_batch.eq(targets).sum().item()  
            
    if individual:
        loss_list = torch.concatenate(loss_list)
        diff_list = torch.concatenate(diff_list)
        return {'y': loss_list.cpu(), 
                'y_stats' :{
                    'mean' : loss_list.mean().item(),
                    'var'  : loss_list.var().item(),
                    'max'  : loss_list.max().item(),
                    'min'  : loss_list.min().item()},
                'y_add1': diff_list.cpu(), 
                'y_add1_stats' :{
                    'mean' : diff_list.mean().item(),
                    'var'  : diff_list.var().item(),
                    'max'  : diff_list.max().item(),
                    'min'  : diff_list.min().item()}
                }

    else:
        return {'y': loss / total, 'y_add1': correct / total} 

def get_imagemodel_hyperparameters(net_type='cnn'):
    depth, widen_factor = 0, 0

    nettype_list = net_type.split('-')
    net_name = nettype_list[0]
    if 'wrn' in net_type:
        if len(nettype_list) > 1:
            _, depth, widen_factor = nettype_list
            depth = int(depth)
            widen_factor = int(widen_factor)
        else:
            depth = 16
            widen_factor = 1  
    elif 'vgg' in net_type: 
        if len(nettype_list) > 1:
            _, depth = nettype_list
            depth = int(depth) 
        else:
            depth = 11 
    elif 'resnet' in net_type: 
        if len(nettype_list) > 1:
            _, depth = nettype_list
            depth = int(depth) 
        else:
            depth = 18 
    elif 'preactrn' in net_type: 
        if len(nettype_list) > 1:
            _, depth = nettype_list
            depth = int(depth) 
        else:
            depth = 18 
    else:
        raise ValueError("net_type={} is not supported".format(net_type))
    return net_name, depth, widen_factor

def get_imgemb_module(net_name):
    return IMGEMB_FACTORY[net_name]
    
# def get_model(net_type:str='cnn', num_classes:int=10, rgb:bool=False, dropout_rate:float=0.):
#     image_channels = 3 if rgb else 1
#     depth, widen_factor = get_imagemodel_hyperparameters(net_type)
#     if net_type == 'cnn':
#         return CNN(
#             dropout_rate=dropout_rate,
#             num_classes=num_classes,
#             num_input_channels=image_channels
#         )    
#     elif 'wrn' in net_type:
#         return Wide_ResNet(
#             depth=depth, 
#             widen_factor=widen_factor, 
#             dropout_rate=dropout_rate, 
#             num_classes=num_classes,
#             num_input_channels=image_channels
#         )
#     elif net_type == 'lenet':
#         return LeNet(
#             dropout_rate=dropout_rate, 
#             num_classes=num_classes,
#             num_input_channels=image_channels
#         ) 
#     elif 'vgg' in net_type:
#         return VGG(
#             depth=depth, 
#             dropout_rate=dropout_rate, 
#             num_classes=num_classes,
#             num_input_channels=image_channels
#         ) 
#         # base_opt = optim.SGD(net.parameters(), lr=0.1)
#         # opt = torchcontrib.optim.SWA(base_opt, swa_start=10, swa_freq=5, swa_lr=0.05)
#     elif 'resnet' in net_type:
#         return ResNet(
#             depth=depth, 
#             dropout_rate=dropout_rate, 
#             num_classes=num_classes,
#             num_input_channels=image_channels
#         ) 
#     elif 'preactrn' in net_type:
#         return PreActResNet(
#             depth=depth, 
#             dropout_rate=dropout_rate, 
#             num_classes=num_classes,
#             num_input_channels=image_channels
#         )
#     else:
#         raise ValueError("net_type={} is not supported".format(net_type))

 