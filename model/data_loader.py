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

class ArchiStyleDataset(Dataset[Tuple[Tensor, int]]):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    def __init__(self, data_dir: str, transform: transforms.Compose):
        """
        Store the filenames of the jpgs to use. Specifies transforms to apply on images.

        Args:
            data_dir: (string) directory containing the dataset
            transform: (torchvision.transforms) transformation to apply on image
            
        Raises:
            FileNotFoundError: If the data directory does not exist.
        """
        super().__init__()
        data_path = Path(data_dir)
        if not data_path.is_dir():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        self.filenames: List[Path] = sorted(
            list(data_path.glob("*.jpg"))
            + list(data_path.glob("*.jpeg"))
            + list(data_path.glob("*.png"))
        )
        if not self.filenames:
            logger.warning(f"No image files found in {data_dir}")

        # self.labels = [int(os.path.split(filename)[-1][0]) for filename in self.filenames]
        self.labels: List[int] = [int(f.name[0]) for f in self.filenames]
        self.transform = transform
        logger.info(f"Loaded {len(self.filenames)} images from {data_dir}")


    def __len__(self) -> int:
        """Return the total number of images in the dataset."""
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            image: (Tensor) transformed image
            label: (int) corresponding label of image
        """
        image_path = self.filenames[idx]
        try:
            image = Image.open(image_path).convert("RGB")
            transformed_image = self.transform(image)
            return transformed_image, self.labels[idx]
        except (IOError, UnidentifiedImageError) as e:
            logger.error(f"Error loading image {image_path}: {e}")
            raise


def fetch_dataloader(types: List[str], data_dir: str, params: DataLoaderParams) -> Dict[str, DataLoader[Tuple[Tensor, int]]]:
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders: Dict[str, DataLoader[Tuple[Tensor, int]]] = {}
    data_path = Path(data_dir)

    for split in ['train', 'val', 'test']:
        if split in types:
            split_path = data_path / split
            logger.info(f"Creating DataLoader for '{split}' split from {split_path}")

            # use the train_transformer if training data, else use eval_transformer without random flip
            if split == 'train':
                transformer = train_transformer
                shuffle = True
            else:
                transformer = eval_transformer
                shuffle = False
            
            dataset = ArchiStyleDataset(str(split_path), transformer)
            dl = DataLoader(dataset, batch_size=params.batch_size, shuffle=shuffle, num_workers=params.num_workers, pin_memory=params.cuda)
            dataloaders[split] = dl

    return dataloaders
