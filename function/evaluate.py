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
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', required=True,
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', required=True,
                    help="Directory containing params.json")
parser.add_argument('--model_name', required=True, help="Model name, e.g. alexnet, vgg19, resnet50, or densenet121")
parser.add_argument('--pretrained', action='store_true', help="Use pre-trained model weights for transfer learning")
parser.add_argument('--gpu_id', type=int, default=0, help="GPU ID to use for evaluation")
parser.add_argument('--save_confusion_matrix', action='store_true', help="Save confusion matrix plot")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")


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
    all_labels = []
    all_preds = []

    # compute metrics over the dataset
    with torch.no_grad():
        for data_batch, labels_batch in dataloader:

            # move to GPU if available
            data_batch, labels_batch = data_batch.to(device), labels_batch.to(device)

            # compute model output
            output_batch = model(data_batch)
            loss = loss_fn(output_batch, labels_batch)

            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch_np = output_batch.data.cpu().numpy()
            labels_batch_np = labels_batch.data.cpu().numpy()

            # get predictions
            preds_batch = np.argmax(output_batch_np, axis=1)
            all_preds.extend(preds_batch.tolist())
            all_labels.extend(labels_batch_np.tolist())


            # compute all metrics on this batch
            summary_batch = {metric: metrics[metric](output_batch_np, labels_batch_np)
                             for metric in metrics}
            summary_batch['loss'] = loss.item()
            summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric]
                                     for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean, all_preds, all_labels


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
    if use_cuda:
        torch.cuda.manual_seed(230)

    # Get the logger
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")
    
    # Add validation to ensure model-dataset compatibility
    model_experiment = Path(args.model_dir).parts[-2]  # e.g., "experiments-v1"
    dataset_version = Path(args.data_dir).name  # e.g., "ArchiStyle-v1"
    
    expected_dataset = "ArchiStyle-v1" if "v1" in model_experiment else "ArchiStyle-v2"
    if dataset_version != expected_dataset:
        logging.warning(f"⚠️  DATASET MISMATCH DETECTED!")
        logging.warning(f"   Model from: {model_experiment}")
        logging.warning(f"   Dataset: {dataset_version}")
        logging.warning(f"   Expected: {expected_dataset}")
        logging.warning(f"   This may result in poor performance!")

    # fetch dataloaders
    data_loader_params = data_loader.DataLoaderParams(
        batch_size=int(params.batch_size),
        num_workers=int(params.num_workers),
        cuda=use_cuda
    )
    # change to test if needed
    dataloaders = data_loader.fetch_dataloader(['val'], args.data_dir, data_loader_params)
    test_dl = dataloaders['val']

    logging.info("- done.")

    # Define the model
    model = net.get_model(args.model_name, pretrained=args.pretrained).to(device)
    
    # Verify model architecture matches the saved weights
    checkpoint_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            try:
                # Try to load the state dict to verify compatibility
                model.load_state_dict(checkpoint['state_dict'])
                logging.info(f"✅ Model architecture ({args.model_name}) matches saved weights")
            except Exception as e:
                logging.error(f"❌ MODEL ARCHITECTURE MISMATCH!")
                logging.error(f"   Trying to load {args.model_name} weights into {args.model_name} model")
                logging.error(f"   Error: {str(e)}")
                logging.error(f"   This explains the poor performance!")
                raise ValueError(f"Model architecture mismatch: {str(e)}")
    
    loss_fn = net.loss_fn
    metrics = net.metrics

    logging.info("Starting evaluation")

    # Reload weights from the saved file (this will work now that we've verified compatibility)
    utils.load_checkpoint(checkpoint_path, model)

    # Evaluate
    test_metrics, all_preds, all_labels = evaluate(model, loss_fn, test_dl, metrics, params, device)
    save_path = os.path.join(
        args.model_dir, "metrics_test_{}.json".format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)
    
    # Log final results with context
    logging.info(f"📊 EVALUATION RESULTS:")
    logging.info(f"   Model: {args.model_name} from {args.model_dir}")
    logging.info(f"   Dataset: {args.data_dir}")
    logging.info(f"   Samples: {len(test_dl.dataset)}")
    for metric, value in test_metrics.items():
        logging.info(f"   {metric}: {value:.4f}")
    
    if args.save_confusion_matrix:
        logging.info("Generating and saving confusion matrices...")
        # Get class names from the dataset
        class_names_en = test_dl.dataset.classes
        
        # Read Chinese class names
        class_names_zh_path = Path(args.data_dir).parent / 'ArchiStyle-v1' / 'class_names_zh.txt'
        class_names_zh = []
        if class_names_zh_path.exists():
            with open(class_names_zh_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # Extract Chinese name from "0 川派" format
                    parts = line.strip().split(' ', 1)
                    if len(parts) == 2:
                        class_names_zh.append(parts[1])
        else:
            logging.warning(f"Chinese class names file not found at {class_names_zh_path}")
            class_names_zh = class_names_en  # Fallback to English names
        
        # Compute confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Convert to percentages (normalize by row to show percentage of true class)
        # Add small epsilon to avoid division by zero
        row_sums = cm.sum(axis=1)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        cm_percentage = cm.astype('float') / row_sums[:, np.newaxis] * 100
        
        # Create images directory
        images_dir = model_dir / 'images'
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot English confusion matrix with percentages
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_percentage, annot=True, fmt='.1f', cmap='Blues', 
                   xticklabels=class_names_en, yticklabels=class_names_en,
                   cbar_kws={'label': 'Percentage (%)'})
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix (%) - {args.model_name} on {Path(args.data_dir).name}\nAccuracy: {test_metrics["accuracy"]:.1%}')
        plt.tight_layout()
        
        # Save English version
        cm_en_save_path = images_dir / 'confusion_matrix_en.png'
        plt.savefig(cm_en_save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"English confusion matrix saved to {cm_en_save_path}")
        
        # Plot Chinese confusion matrix with percentages
        plt.figure(figsize=(10, 8))
        
        # --- Robust Font Handling for Chinese Characters ---
        import warnings
        import matplotlib.font_manager as fm
        
        # List of common fonts that support Chinese characters
        font_list = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS', 'Heiti TC', 'PingFang SC']
        
        # Find available fonts on the system
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        # Find the best available font
        found_font = None
        for font in font_list:
            if font in available_fonts:
                found_font = font
                logging.info(f"Using font '{font}' for Chinese characters.")
                break
        
        if found_font:
            # Suppress font warnings temporarily
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                plt.rcParams['font.sans-serif'] = [found_font, 'DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False
                
                # Test if the font actually works by creating a small test plot
                test_fig, test_ax = plt.subplots()
                test_ax.text(0.5, 0.5, '测试', fontsize=12)
                plt.close(test_fig)
                
                # Create the actual confusion matrix
                sns.heatmap(cm_percentage, annot=True, fmt='.1f', cmap='Blues', 
                           xticklabels=class_names_zh, yticklabels=class_names_zh,
                           cbar_kws={'label': '百分比 (%)'})
                plt.xlabel('预测')
                plt.ylabel('真实')
                plt.title(f'混淆矩阵 (%) - {args.model_name} on {Path(args.data_dir).name}\n准确率: {test_metrics["accuracy"]:.1%}')
                
                use_chinese = True
        else:
            logging.warning("No suitable Chinese font found. Using English labels for Chinese confusion matrix.")
            # Fallback to English labels but keep Chinese title elements where possible
            sns.heatmap(cm_percentage, annot=True, fmt='.1f', cmap='Blues', 
                       xticklabels=class_names_en, yticklabels=class_names_en,
                       cbar_kws={'label': 'Percentage (%)'})
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix (%) - {args.model_name} on {Path(args.data_dir).name}\nAccuracy: {test_metrics["accuracy"]:.1%}')
            use_chinese = False
        
        plt.tight_layout()
        
        # Save Chinese version (or English fallback)
        cm_zh_save_path = images_dir / ('confusion_matrix_zh.png' if use_chinese else 'confusion_matrix_zh_fallback.png')
        plt.savefig(cm_zh_save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        if use_chinese:
            logging.info(f"Chinese confusion matrix saved to {cm_zh_save_path}")
        else:
            logging.info(f"Chinese confusion matrix (English fallback) saved to {cm_zh_save_path}")
        
        # Reset matplotlib font settings to default
        plt.rcParams['font.sans-serif'] = plt.rcParamsDefault['font.sans-serif']
        plt.rcParams['axes.unicode_minus'] = plt.rcParamsDefault['axes.unicode_minus']
