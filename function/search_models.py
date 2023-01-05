"""Peform hyperparemeters search"""

import argparse
import os
from subprocess import check_call
import sys

import utils


PYTHON = sys.executable
parser = argparse.ArgumentParser()
# baseline 
# parser.add_argument('--parent_dir', default='experiments1/baseline_model', help='Directory containing params.json')
# TL basedline
# parser.add_argument('--parent_dir', default='experiments-baseline/baseline_model', help='Directory containing params.json')
# TL basedline + DA
parser.add_argument('--parent_dir', default='../experiments-baseline/DA', help='Directory containing params.json')
# TL basedline + LDA
# parser.add_argument('--parent_dir', default='experiments-baseline/LDA', help='Directory containing params.json')
parser.add_argument('--data_dir', default='../data/256x256_ArchiStyle', help="Directory containing the dataset")


def launch_training_job(parent_dir, data_dir,model_name, job_name, params):
    """Launch training of the model with a set of hyperparameters in parent_dir/job_name

    Args:
        model_dir: (string) directory containing config, weights and log
        data_dir: (string) directory containing the dataset
        params: (dict) containing hyperparameters
    """
    # Create a new folder in parent_dir with unique_name "job_name"
    model_dir = os.path.join(parent_dir, job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Write parameters in json file
    json_path = os.path.join(model_dir, 'params.json')
    params.save(json_path)

    # Launch training with this config
    cmd = "{python} train.py --model_dir={model_dir} --data_dir {data_dir} --model_name {model_name}".format(python=PYTHON, model_dir=model_dir,
                                                                                   data_dir=data_dir, model_name=model_name)
    print(cmd)
    check_call(cmd, shell=True)


if __name__ == "__main__":
    # Load the "reference" parameters from parent_dir json file
    args = parser.parse_args()
    json_path = os.path.join(args.parent_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Perform hypersearch over one parameter
    # baseline_models = ['alexnet','vggnet','resnet','densenet']
    baseline_models = ['alexnet','vggnet','resnet']
    # baseline_models = ['vggnet']
    # baseline_models = ['alexnet','resnet','densenet']
    # baseline_models = ['densenet']

    for baseline_model in baseline_models:
        # Modify the relevant parameter in params
        params.model_name = baseline_model

        # Launch job (name has to be unique)
        job_name = "baseline_model_{}".format(baseline_model)
        launch_training_job(args.parent_dir, args.data_dir,baseline_model, job_name, params)
