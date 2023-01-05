"""Split the dataset into train/val/test and resize images to 256x250

The SIGNS dataset comes into the following format:
    train_signs/
        0_28.jpg
        ...
    test_signs/
        0_72.jpg
        ...

Original images have diff. size.
Resizing to (256,256) reduces the dataset size, and loading smaller images
makes training faster.

We already have a test set created, so we only need to split "train_signs" into train and val sets.
Because we don't have a lot of images and we want that the statistics on the val set be as
representative as possible, we'll take 20% of "train_signs" as val set.
"""

import argparse
import random
import os

from PIL import Image
from tqdm import tqdm

SIZE = 256

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='../data/ArchiStyle', help="Directory with the dataset")
parser.add_argument('--output_dir', default='../data/256x256_ArchiStyle', help="Where to write the new data")


def resize_and_save(filename, output_dir, size=SIZE):
    """Resize the image contained in `filename` and save it to the `output_dir`"""
    image = Image.open(filename)
    # issue: https://stackoverflow.com/questions/48248405/cannot-write-mode-rgba-as-jpeg/48248432
    image = image.convert('RGB')
    # Use bilinear interpolation instead of the default "nearest neighbor" method
    image = image.resize((size, size), Image.BILINEAR)
    image.save(os.path.join(output_dir, filename.split('/')[-1]))

def replace_windows_path(names):
    names_new=[]
    for name in names:
        if "\\" in name:
            name = name.replace('\\','/')
            names_new.append(name)
    return names_new

EXTENSION_IMAGE = (".png",".jpg",".jpeg")
if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    # Define the data directories
    train_data_dir = os.path.join(args.data_dir, 'train')
    test_data_dir = os.path.join(args.data_dir, 'test')

    # Get the filenames in each directory (train and test)
    filenames = os.listdir(train_data_dir)
    filenames = [os.path.join(train_data_dir, f) for f in filenames if f.endswith(EXTENSION_IMAGE)]
    # handle windows path
    # filenames = replace_windows_path(filenames)

    test_filenames = os.listdir(test_data_dir)
    test_filenames = [os.path.join(test_data_dir, f) for f in test_filenames if f.endswith(EXTENSION_IMAGE)]
    # handle windows path
    # test_filenames = replace_windows_path(test_filenames)

    # Split the images in 'train_signs' into 80% train and 20% val
    # Make sure to always shuffle with a fixed seed so that the split is reproducible
    random.seed(230)
    filenames.sort()
    random.shuffle(filenames)

    split = int(0.8 * len(filenames))
    train_filenames = filenames[:split]
    val_filenames = filenames[split:]

    filenames = {'train': train_filenames,
                 'val': val_filenames,
                 'test': test_filenames}

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print("Warning: output dir {} already exists".format(args.output_dir))

    # Preprocess train, val and test
    for split in ['train', 'val', 'test']:
        output_dir_split = os.path.join(args.output_dir, '{}'.format(split))
        if not os.path.exists(output_dir_split):
            os.mkdir(output_dir_split)
        else:
            print("Warning: dir {} already exists".format(output_dir_split))

        print("Processing {} data, saving preprocessed data to {}".format(split, output_dir_split))
        for filename in tqdm(filenames[split]):
            resize_and_save(filename, output_dir_split, size=SIZE)

    print("Done building dataset")
