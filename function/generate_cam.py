"""
Generates and saves Class Activation Maps (CAMs) for a trained model.

This script is a refactored and generalized version of the logic from
grad-cam/main_densenet-TL-AD.py. It can be used with different models
and datasets.
"""
import argparse
import logging
import os
import sys
from pathlib import Path
import random

import numpy as np
import torch
import cv2
from PIL import Image

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import utils
import model.net as net
import model.data_loader as data_loader
from torchvision import transforms

parser = argparse.ArgumentParser(description="Generate Class Activation Maps (CAMs)")
parser.add_argument('--data_dir', default='data/ArchiStyle-v2',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default='best',
                    help="Name of the file in --model_dir containing weights to load")
parser.add_argument('--model_name', default='alexnet',
                    help="Model name, e.g. alexnet, vgg19, resnet50, or densenet121")
parser.add_argument('--pretrained', action='store_true',
                    help="Use pre-trained model weights (for model definition)")
parser.add_argument('--gpu_id', type=int, default=0,
                    help="GPU ID to use for evaluation")
parser.add_argument('--num_images_per_class', type=int, default=5,
                    help="Number of images to generate CAMs for per class")


class GradCAM:
    """
    Calculates Grad-CAM for a given model.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handles = []

    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, grad):
        self.gradients = grad.detach()

    def __call__(self, x, class_idx=None):
        # Reset state
        self.gradients = None
        self.activations = None
        
        # Clear any existing handles
        for handle in self.handles:
            handle.remove()
        self.handles = []
        
        # Set model to eval mode but enable gradients
        self.model.eval()
        
        # Enable gradients for input
        x = x.clone().detach().requires_grad_(True)
        
        # Register forward hook
        forward_handle = self.target_layer.register_forward_hook(self.save_activations)
        self.handles.append(forward_handle)
        
        # Forward pass
        output = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        # Get the score for the target class
        score = output[0, class_idx]
        
        # Register backward hook on activations if they require grad
        if self.activations is not None and self.activations.requires_grad:
            backward_handle = self.activations.register_hook(self.save_gradients)
            self.handles.append(backward_handle)
        
        # Backward pass
        self.model.zero_grad()
        score.backward(retain_graph=True)

        # Clean up handles
        for handle in self.handles:
            handle.remove()
        self.handles = []

        # Check if gradients and activations were captured
        if self.gradients is None or self.activations is None:
            # Fallback: try a simpler approach without hooks
            return self._fallback_gradcam(x, class_idx)

        # Convert to numpy
        gradients = self.gradients.cpu().numpy()[0]
        activations = self.activations.detach().cpu().numpy()[0]

        # Pool gradients across channels
        pooled_gradients = np.mean(gradients, axis=(1, 2))

        # Weight the channels by corresponding gradients
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]

        # Average the channels of the activations
        heatmap = np.mean(activations, axis=0)
        # ReLU on top of the heatmap
        heatmap = np.maximum(heatmap, 0)
        # Normalize the heatmap
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap)
            
        return heatmap, class_idx
    
    def _fallback_gradcam(self, x, class_idx):
        """Fallback method that computes gradients using a different approach."""
        # This approach uses the input gradients to approximate the effect
        x = x.clone().detach().requires_grad_(True)
        
        # Get activations without detaching
        activations = None
        def get_activations(module, input, output):
            nonlocal activations
            activations = output
        
        handle = self.target_layer.register_forward_hook(get_activations)
        
        try:
            output = self.model(x)
            score = output[0, class_idx]
            
            # Compute gradients of score w.r.t. input
            input_grad = torch.autograd.grad(score, x, retain_graph=True)[0]
            
            # Use a simple approximation: create heatmap from input gradients
            # This is not as accurate as true GradCAM but should work as fallback
            grad_magnitude = torch.mean(torch.abs(input_grad), dim=1, keepdim=True)
            heatmap = grad_magnitude.squeeze().detach().cpu().numpy()
            
            # Resize to match expected output size (assuming square activations)
            if activations is not None:
                target_size = activations.shape[-1]  # Assume square
                import cv2
                heatmap = cv2.resize(heatmap, (target_size, target_size))
            
            # Normalize the heatmap
            if np.max(heatmap) > 0:
                heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
                
            return heatmap, class_idx
            
        except Exception as e:
            # If even the fallback fails, return a zero heatmap
            logging.warning(f"Fallback GradCAM also failed: {e}. Returning zero heatmap.")
            if activations is not None:
                target_size = activations.shape[-1]
                heatmap = np.zeros((target_size, target_size))
            else:
                heatmap = np.zeros((224, 224))  # Default size
            return heatmap, class_idx
            
        finally:
            handle.remove()

def get_target_layer(model, model_name):
    if 'resnet' in model_name:
        return model.layer4
    elif 'densenet' in model_name:
        return model.features
    elif 'alexnet' in model_name or 'vgg' in model_name:
        # Filter for Conv2d layers and get the last one
        conv_layers = [layer for layer in model.features if isinstance(layer, torch.nn.Conv2d)]
        if not conv_layers:
            raise ValueError("Could not find a Conv2d layer in model.features")
        return conv_layers[-1]
    else:
        raise ValueError(f"Model name {model_name} not supported for Grad-CAM")


def main():
    args = parser.parse_args()
    model_dir = Path(args.model_dir)
    json_path = model_dir / 'params.json'
    assert json_path.is_file(), f"No json configuration file found at {json_path}"
    params = utils.Params(json_path)

    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu_id}" if use_cuda else "cpu")
    params.device = device
    params.cuda = use_cuda
    
    utils.set_logger(model_dir / 'generate_cam.log')
    logging.info("Loading model and data...")
    
    # Define the model
    model = net.get_model(args.model_name, pretrained=args.pretrained).to(device)
    
    # Reload weights
    utils.load_checkpoint(model_dir / (args.restore_file + '.pth.tar'), model)
    model.eval()

    # fetch dataloaders
    data_loader_params = data_loader.DataLoaderParams(
        batch_size=int(params.batch_size),
        num_workers=int(params.num_workers),
        cuda=params.cuda
    )
    
    # Get the validation dataset
    dataloaders = data_loader.fetch_dataloader(['val'], args.data_dir, data_loader_params)
    val_dataloader = dataloaders.get('val')
    if val_dataloader is None:
        logging.error("Could not find the validation dataloader. Make sure 'val' data is available.")
        sys.exit(1)

    val_dataset = val_dataloader.dataset
    class_names = val_dataset.classes

    # Create output directory
    cam_output_dir = model_dir / 'images' / 'cams'
    cam_output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving CAM images to: {cam_output_dir}")

    # Group images by class
    class_to_images = {i: [] for i in range(len(class_names))}
    for i in range(len(val_dataset.image_filenames)):
        class_idx = val_dataset.labels[i]
        image_path = val_dataset.data_dir / val_dataset.image_filenames[i]
        class_to_images[class_idx].append(str(image_path))

    # Initialize Grad-CAM
    target_layer = get_target_layer(model, args.model_name)
    grad_cam = GradCAM(model, target_layer)

    # Image transformations
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    for class_idx, class_name in enumerate(class_names):
        logging.info(f"Processing class: {class_name}")
        image_paths = random.sample(class_to_images[class_idx], min(args.num_images_per_class, len(class_to_images[class_idx])))

        for img_path in image_paths:
            try:
                # Load and preprocess image
                original_img = cv2.imread(img_path)
                if original_img is None:
                    logging.warning(f"Could not read image {img_path}, skipping.")
                    continue
                
                original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(original_img_rgb)
                input_tensor = data_transforms(img_pil).unsqueeze(0).to(device)

                # Generate heatmap
                heatmap, predicted_class_idx = grad_cam(input_tensor)

                # Superimpose heatmap on original image
                heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed_img = heatmap * 0.4 + original_img

                # Save the image
                predicted_class_name = class_names[predicted_class_idx]
                img_name = Path(img_path).stem
                save_path = cam_output_dir / f"{class_name}_{img_name}_pred-{predicted_class_name}_cam.png"
                cv2.imwrite(str(save_path), superimposed_img)
            except Exception as e:
                logging.error(f"Failed to process {img_path}: {e}")
                
    logging.info("Finished generating CAMs.")


if __name__ == '__main__':
    main() 