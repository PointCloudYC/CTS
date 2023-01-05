import random
import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# data augmentation
from model.auto_augment import AutoAugment, Cutout

id_to_names={
    "0":"chuan",
    "1":"jin",
    "2":"jing",
    "3":"min",
    "4":"su",
    "5":"wan"
}

IMAGE_SIZE =256
IMAGE_NETWORK_SIZE=224
# For more tranforms, check https://pytorch.org/docs/stable/torchvision/transforms.html--yc
# for insights, check https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/5.2-using-convnets-with-small-datasets.ipynb
train_transformer = transforms.Compose([
    transforms.Resize(IMAGE_NETWORK_SIZE),  # resize the image to 64x64 (remove if images are already 64x64)

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
eval_transformer = transforms.Compose([
    transforms.Resize(IMAGE_NETWORK_SIZE),  # resize the image to 64x64 (remove if images are already 64x64)
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
        self.filenames = os.listdir(data_dir)
        self.filenames = [os.path.join(data_dir, f) for f in self.filenames if f.endswith('.jpg')]

        self.labels = [int(os.path.split(filename)[-1][0]) for filename in self.filenames]
        self.transform = transform

    def __len__(self):
        # return size of dataset
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            image: (Tensor) transformed image
            label: (int) corresponding label of image
        """
        image = Image.open(self.filenames[idx])  # PIL image
        image = self.transform(image)
        return image, self.labels[idx]


def fetch_dataloader(types, data_dir, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}

    for split in ['train', 'val', 'test']:
        if split in types:
            path = os.path.join(data_dir, "{}".format(split))

            # use the train_transformer if training data, else use eval_transformer without random flip
            if split == 'train':
                dl = DataLoader(ArchiStyleDataset(path, train_transformer), batch_size=params.batch_size, shuffle=True,
                                        num_workers=params.num_workers,
                                        pin_memory=params.cuda)
            else:
                dl = DataLoader(ArchiStyleDataset(path, eval_transformer), batch_size=params.batch_size, shuffle=False,
                                num_workers=params.num_workers,
                                pin_memory=params.cuda)

            dataloaders[split] = dl

    return dataloaders
