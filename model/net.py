"""Defines the neural network, losss function and metrics"""

from typing import Dict, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from sklearn.metrics import precision_score, recall_score, f1_score

NUM_CLASSES: int = 6


def get_model(model_name: str, pretrained: bool = True) -> nn.Module:
    """
    Factory function to create a pre-trained model for transfer learning.

    This function fetches a specified model from torchvision.models, optionally
    with pre-trained weights, and replaces its final classifier layer with a new
    one suitable for the number of classes in this project.

    Args:
        model_name (str): The name of the model to create (e.g., 'alexnet', 
                          'vgg19', 'densenet121', 'resnet50').
        pretrained (bool): If True, loads weights pre-trained on ImageNet.

    Returns:
        nn.Module: The configured PyTorch model.

    Raises:
        ValueError: If the model_name is not supported.
    """
    model: nn.Module
    if model_name == 'alexnet':
        model = models.alexnet(pretrained=pretrained)
    elif model_name == 'vgg19':
        model = models.vgg19(pretrained=pretrained)
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=pretrained)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
    else:
        raise ValueError(f"Model {model_name} not supported.")

    # Freeze all layers if using a pre-trained model
    if pretrained:
        for param in model.parameters():
            param.requires_grad = False
    
    # Replace the final classifier. New layers have requires_grad=True by default.
    if model_name == 'alexnet':
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(num_ftrs, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, NUM_CLASSES),
        )
    elif model_name == 'vgg19':
        num_ftrs = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, NUM_CLASSES),
        )
    elif model_name == 'densenet121':
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, NUM_CLASSES)
    elif model_name == 'resnet50':
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    
    return model

def loss_fn(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute the cross entropy loss given outputs and labels.

    Args:
        outputs (torch.Tensor): The output of the model, of shape (batch_size, num_classes).
        labels (torch.Tensor): The true labels, of shape (batch_size,).

    Returns:
        torch.Tensor: The computed cross entropy loss.
    """
    return F.cross_entropy(outputs, labels)


def accuracy(outputs: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute the accuracy, given the outputs and labels for all images.

    Args:
        outputs (np.ndarray): Log softmax output of the model of shape (batch_size, num_classes).
        labels (np.ndarray): True labels, of shape (batch_size,).

    Returns:
        float: The accuracy in the range [0, 1].
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs == labels) / float(labels.size)


def precision(outputs: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute the weighted precision score.

    Args:
        outputs (np.ndarray): Log softmax output of the model.
        labels (np.ndarray): True labels.

    Returns:
        float: The weighted precision score.
    """
    outputs = np.argmax(outputs, axis=1)
    return precision_score(labels, outputs, average='weighted', zero_division=0)


def recall(outputs: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute the weighted recall score.

    Args:
        outputs (np.ndarray): Log softmax output of the model.
        labels (np.ndarray): True labels.

    Returns:
        float: The weighted recall score.
    """
    outputs = np.argmax(outputs, axis=1)
    return recall_score(labels, outputs, average='weighted', zero_division=0)


def f1score(outputs: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute the weighted F1 score.

    Args:
        outputs (np.ndarray): Log softmax output of the model.
        labels (np.ndarray): True labels.

    Returns:
        float: The weighted F1 score.
    """
    outputs = np.argmax(outputs, axis=1)
    return f1_score(labels, outputs, average='weighted', zero_division=0)

# A dictionary mapping metric names to their respective functions.
metrics: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1score': f1score,
}
