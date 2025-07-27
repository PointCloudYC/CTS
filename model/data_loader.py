import os
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, UnidentifiedImageError
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# data augmentation
from model.auto_augment import AutoAugment, Cutout


logger = logging.getLogger(__name__)


ID_TO_NAMES: Dict[str, str] = {
    "0":"chuan",
    "1":"jin",
    "2":"jing",
    "3":"min",
    "4":"su",
    "5":"wan"
}

# Define class names based on your dataset's convention
CLASS_NAMES = [ID_TO_NAMES[str(i)] for i in range(len(ID_TO_NAMES))]


@dataclass
class DataLoaderParams:
    """Data loader parameters."""
    batch_size: int
    num_workers: int
    cuda: bool

IMAGE_SIZE: int = 256
IMAGE_NETWORK_SIZE: int = 224

# For more tranforms, check https://pytorch.org/docs/stable/torchvision/transforms.html--yc
# for insights, check https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/5.2-using-convnets-with-small-datasets.ipynb
train_transformer: transforms.Compose = transforms.Compose([
    transforms.Resize((IMAGE_NETWORK_SIZE, IMAGE_NETWORK_SIZE)),  # resize the image to 224x224

    # https://github.com/ildoonet/pytorch-randaugment
    # Add RandAugment with N, M(hyperparameter)
    # transform_train.transforms.insert(0, RandAugment(N, M))

    # transforms.RandomAffine(degrees, translate=None, scale=None, shear=None, resample=0, fillcolor=0),
    # transforms.RandomRotation(degrees=(-40, 40)),
    # transforms.CenterCrop(IMAGE_NETWORK_SIZE),

    # my data augmentation
    # transforms.RandomCrop(IMAGE_NETWORK_SIZE, padding=4), # not use ,because the model shape is fixed written, can not use
    # transforms.RandomHorizontalFlip(p=0.5),


    # transforms.RandomVerticalFlip(p=0.5),
    # transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0, hue=0),
    # AutoAugment(), # AutoAug
    transforms.ToTensor(),
    # transforms.Normalize([0, 0, 0], [1, 1, 1])],
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    # transforms.RandomErasing(),
    ])# transform it into a torch tensor

# loader for evaluation, no horizontal flip
eval_transformer: transforms.Compose = transforms.Compose([
    transforms.Resize((IMAGE_NETWORK_SIZE, IMAGE_NETWORK_SIZE)),  # resize the image to 224x224
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    # transforms.Normalize([0, 0, 0], [1, 1, 1]),
    ])  # transform it into a torch tensor

class ArchiStyleDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """

    def __init__(self, data_dir, transform):
        """
        Store the filenames of the jpgs to use. Specifies transforms to apply on images.

        Args:
            data_dir: (string) directory containing the dataset
            transform: (torchvision.transforms) transformation to apply on image
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.image_filenames = [f for f in os.listdir(self.data_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        self.labels = [int(f.split('_')[0]) for f in self.image_filenames]
        self.classes = CLASS_NAMES


    def __len__(self):
        # return size of dataset
        return len(self.image_filenames)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            image: (Tensor) transformed image
            label: (int) corresponding label of image
        """
        image_path = self.data_dir / self.image_filenames[idx]
        try:
            image = Image.open(image_path).convert("RGB")
            transformed_image = self.transform(image)
            return transformed_image, self.labels[idx]
        except (IOError, UnidentifiedImageError) as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return a dummy tensor and label, or handle appropriately
            # This part depends on how you want to handle corrupted images
            # For now, we re-raise the exception.
            raise


def fetch_dataloader(types: List[str], data_dir: str, params: DataLoaderParams) -> Dict[str, DataLoader]:
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders: Dict[str, DataLoader] = {}
    data_path = Path(data_dir)

    for split in ['train', 'val', 'test']:
        if split in types:
            split_path = data_path / split
            logger.info(f"Creating DataLoader for '{split}' split from {split_path}")
            if not split_path.is_dir():
                logger.warning(f"Directory not found for split '{split}': {split_path}")
                continue

            # use the train_transformer if training data, else use eval_transformer
            transformer = train_transformer if split == 'train' else eval_transformer
            shuffle = split == 'train'
            
            dataset = ArchiStyleDataset(str(split_path), transformer)
            
            # Check if dataset is empty
            if not len(dataset):
                logger.warning(f"No images found in {split_path}, skipping dataloader creation.")
                continue

            dl = DataLoader(dataset, batch_size=params.batch_size, shuffle=shuffle, num_workers=params.num_workers, pin_memory=params.cuda)
            dataloaders[split] = dl

    return dataloaders
