"""Evaluates the model"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import numpy as np
import torch
from torch.autograd import Variable
import utils
import model.net as net
import model.data_loader as data_loader

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/ArchiStyle-v2',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")
parser.add_argument('--model_name', default='alexnet', help="Model name, e.g. alexnet, vgg19, resnet50, or densenet121")
parser.add_argument('--pretrained', action='store_true', help="Use pre-trained model weights for transfer learning")
parser.add_argument('--gpu_id', type=int, default=0, help="GPU ID to use for evaluation")


def evaluate(model, loss_fn, dataloader, metrics, params, device):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        device: (torch.device) the device to use for evaluation (e.g., 'cuda:0' or 'cpu')
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []

    # compute metrics over the dataset
    with torch.no_grad():
        for data_batch, labels_batch in dataloader:

            # move to GPU if available
            data_batch, labels_batch = data_batch.to(device), labels_batch.to(device)

            # compute model output
            output_batch = model(data_batch)
            loss = loss_fn(output_batch, labels_batch)

            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch = output_batch.data.cpu().numpy()
            labels_batch = labels_batch.data.cpu().numpy()

            # compute all metrics on this batch
            summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                             for metric in metrics}
            summary_batch['loss'] = loss.item()
            summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric]
                                     for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    model_dir = Path(args.model_dir)
    json_path = model_dir / 'params.json'
    assert json_path.is_file(), f"No json configuration file found at {json_path}"
    params = utils.Params(json_path)

    # use GPU if available
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu_id}" if use_cuda else "cpu")
    params.device = device

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # Get the logger
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # fetch dataloaders
    data_loader_params = data_loader.DataLoaderParams(
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        cuda=use_cuda
    )
    dataloaders = data_loader.fetch_dataloader(['test'], args.data_dir, data_loader_params)
    test_dl = dataloaders['test']

    logging.info("- done.")

    # Define the model
    model = net.get_model(args.model_name, pretrained=args.pretrained).to(device)
    loss_fn = net.loss_fn
    metrics = net.metrics

    logging.info("Starting evaluation")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(
        args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    test_metrics = evaluate(model, loss_fn, test_dl, metrics, params, device)
    save_path = os.path.join(
        args.model_dir, "metrics_test_{}.json".format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)
