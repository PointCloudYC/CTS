"""Train the model""" 
import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from tqdm import tqdm
from typing import Callable, Dict
import torch.nn as nn
from torch.utils.data import DataLoader

import utils
import model.net as net
# from model.net import model_factory, model_factory_pretraining
import model.data_loader as data_loader
from evaluate import evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/ArchiStyle-v2', help="Directory containing the dataset")
parser.add_argument('--model_name', default='alexnet', help="Model name, e.g. alexnet, vgg19, resnet50, or densenet121")
parser.add_argument('--pretrained', action='store_true', help="Use pre-trained model weights for transfer learning")
parser.add_argument('--model_dir', default='experiments-v1/alexnet',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None, help="Optional, directory or file containing weights to reload before training")
parser.add_argument('--gpu_id', type=int, default=0, help="GPU ID to use for training")


def train(model: nn.Module, optimizer: optim.Optimizer, loss_fn: Callable, dataloader: DataLoader,
          metrics: Dict[str, Callable], params: utils.Params, device: torch.device) -> Dict[str, float]:
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):
            # move to GPU if available
            train_batch, labels_batch = train_batch.to(device), labels_batch.to(device)
            # convert to torch Variables
            train_batch, labels_batch = Variable(
                train_batch), Variable(labels_batch)

            # compute model output and loss
            # TODO: 32,3,256,256 --> 512,6  wrong! should be 32,6
            output_batch = model(train_batch)
            loss = loss_fn(output_batch, labels_batch)

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()

                # compute all metrics on this batch
                summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                                 for metric in metrics}
                summary_batch['loss'] = loss.item()
                summ.append(summary_batch)

            # update the average loss
            loss_avg.update(loss.item())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric]
                                     for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)

    # print(metrics_mean)
    return metrics_mean

def plot_train_val_curve(history, metric_name, model_dir):
    """plot a metric curve 

    Args:
        history: metric history array, each item is a tuple like (metrics_train, metrics_val); metric_train/val is a dict containing metric and metric values.
        metric_name: (string)metric name , e.g. accuracy, f1score, recall, precision and loss
    """
    metric_train = []
    metric_val = []
    for i in range(len(history)):
        metric_train.append(history[i][0][metric_name])
        metric_val.append(history[i][1][metric_name])

    epochs = range(len(metric_train))
    plt.figure()
    plt.plot(epochs, metric_train, 'b--', label=f'Training {metric_name}', linewidth=2)
    plt.plot(epochs, metric_val, 'g-', label=f'Validation {metric_name}', linewidth=2)
    plt.title(f'Training and validation {metric_name}')
    plt.legend(loc="lower right") # add legend to lower right
    # save the figure
    image_path = os.path.join(model_dir, "images")
    utils.save_fig(f"{metric_name}_curve",image_path)
    plt.show()


def train_and_evaluate(model: nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader,
                       optimizer: optim.Optimizer, loss_fn: Callable, metrics: Dict[str, Callable],
                       params: utils.Params, model_dir: str, device: torch.device, restore_file: str = None):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(
            args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0
    
    # If the model is pretrained, only the classifier's weights are saved.
    # We need to save the full model state dict during this training run.
    if args.pretrained:
        logging.info("Pretrained model detected, will save the full model state_dict.")
        # Replace the model with the full model for saving
        # This is a bit of a workaround to ensure the whole model is saved.
        pass


    history = []
    for epoch in range(int(params.num_epochs)):
        # Run one epoch
        logging.info(f"Epoch {epoch + 1}/{int(params.num_epochs)}")

        # compute number of batches in one epoch (one full pass over the training set)
        train_metrics=train(model, optimizer, loss_fn, train_dataloader, metrics, params, device)

        # Evaluate for one epoch on validation set
        val_metrics, _, _ = evaluate(model, loss_fn, val_dataloader, metrics, params, device)

        # collect the metrics to history 
        history.append((train_metrics,val_metrics))

        val_acc = val_metrics['accuracy']
        is_best = val_acc >= best_val_acc

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=model_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(
                model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(
            model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)

    # plot metric curve during epochs, e.g. accuracy (history=[(train_metrics,val_metrics),... ])
    metric_names =[k for k,v in history[0][0].items()]
    for name in metric_names:
        plot_train_val_curve(history=history,metric_name=name,model_dir=model_dir)


if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    model_dir = Path(args.model_dir)
    params_path = model_dir / 'params.json'

    # Create model directory if it doesn't exist
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load parameters
    if params_path.is_file():
        logging.info(f"Loading parameters from {params_path}")
        params = utils.Params(params_path)
    else:
        logging.info("No params.json found, using default configuration.")
        default_params_path = Path(__file__).parent.parent / 'config' / 'params.json'
        params = utils.Params(default_params_path)
        # Save the default parameters to the model directory for reproducibility
        utils.save_dict_to_json(params.dict, params_path)
        logging.info(f"Saved default parameters to {params_path}")


    # use GPU if available
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu_id}" if use_cuda else "cpu")
    params.device = device

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if use_cuda:
        torch.cuda.manual_seed(230)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders
    data_loader_params = data_loader.DataLoaderParams(
        batch_size=int(params.batch_size),
        num_workers=int(params.num_workers),
        cuda=use_cuda
    )
    dataloaders = data_loader.fetch_dataloader(
        ['train', 'val'], args.data_dir, data_loader_params)
    train_dl = dataloaders['train']
    val_dl = dataloaders['val']

    logging.info("- done.")

    # Define the model and optimizer
    model = net.get_model(args.model_name, pretrained=args.pretrained).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # fetch loss function and metrics
    loss_fn = net.loss_fn
    # loss_fn = torch.nn.NLLLoss
    metrics = net.metrics

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(int(params.num_epochs)))
    train_and_evaluate(model, train_dl, val_dl, optimizer, loss_fn, metrics, params, args.model_dir, device,
                       args.restore_file)
