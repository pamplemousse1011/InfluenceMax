import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from typing import Callable

from model_classifier import LitModel, MultiNet

def train_pl_model(
    model_factory: Callable[[], nn.Module],
    num_models: int,
    train_loader: DataLoader,
    num_gpus: int,
    args,
    test_loader: DataLoader=None,
    validation_loader: DataLoader=None,
):  

    if args.progress_bar:
        progress_bar_refresh_rate=None
    else:
        progress_bar_refresh_rate=0 
        
    multinet = MultiNet(model_factory, num_models)
    litmodel = LitModel(multinet, args.learning_rate)

    kwargs = {
        'max_epochs': args.max_epochs, 
        'min_epochs': args.min_epochs,
        'gpus': num_gpus, 
        'weights_summary': None,
        'progress_bar_refresh_rate': progress_bar_refresh_rate,
        'check_val_every_n_epoch': args.check_val_every_n_epoch,
        'default_root_dir': './logs/'
    }
    if args.early_stopping:
        early_stopping = EarlyStopping('val_acc', patience=args.early_stopping_patience, mode='max')
        trainer = pl.Trainer(
            **kwargs, 
            callbacks=[early_stopping],
        )
    else:
        trainer = pl.Trainer(**kwargs)

    trainer.fit(
        model=litmodel, 
        train_dataloader=train_loader,
        val_dataloaders=validation_loader
    )
    if test_loader is None:
        test_metrics = None
    else:
        test_metrics = trainer.test(test_dataloaders=test_loader)[0]
            
    return multinet, test_metrics



