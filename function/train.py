"""Train the model""" 
import argparse
import logging
import os
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from tqdm import tqdm

import utils
import model.net as net
from model.net import model_factory, model_factory_pretraining
import model.data_loader as data_loader
from evaluate import evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='../data/256x256_ArchiStyle', help="Directory containing the dataset")
# parser.add_argument('--model_dir', default='experiments/baseline_models/baseline_models_alexnet',
#                     help="Directory containing params.json")
parser.add_argument('--model_dir', default='../experiments-baseline/baseline_model/baseline_model_vggnet',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None, help="Optional, directory or file containing weights to reload before training")
parser.add_argument('--model_name', default='vggnet', help="model names, e.g. alexnet, vggnet16, resnet50, or densenet or transfer_learning")  # 'best' or 'train'


def train(model, optimizer, loss_fn, dataloader, metrics, params):
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
            if params.cuda:
                train_batch, labels_batch = train_batch.cuda(
                    non_blocking=True), labels_batch.cuda(non_blocking=True)
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


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn, metrics, params, model_dir,
                       restore_file=None):
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

    history = []
    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train_metrics=train(model, optimizer, loss_fn, train_dataloader, metrics, params)

        # Evaluate for one epoch on validation set
        val_metrics = evaluate(model, loss_fn, val_dataloader, metrics, params)

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
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(
        ['train', 'val'], args.data_dir, params)
    train_dl = dataloaders['train']
    val_dl = dataloaders['val']

    logging.info("- done.")


    # create a model 
    # baseline methods (4 traditional methods + 1 resnet pretraining model)
    # model = model_factory(args.model_name, params, pretrained=False)
    # model = model_factory_pretraining(args.model_name, params, pretrained=False)

    # transfer learning models
    model = model_factory_pretraining(args.model_name, params, pretrained=False)

    # Define the model and optimizer
    # model = net.Net(params).cuda() if params.cuda else net.Net(params)
    model = (model).cuda() if params.cuda else model

    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # fetch loss function and metrics
    loss_fn = net.loss_fn
    # loss_fn = torch.nn.NLLLoss
    metrics = net.metrics

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, train_dl, val_dl, optimizer, loss_fn, metrics, params, args.model_dir,
                       args.restore_file)
