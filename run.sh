#! /bin/bash

# baseline
python function/train.py --model_dir ../experiments/base_model/
python function/evaluate.py --model_dir ../experiments/base_model/

# data aug.
python function/train.py --model_dir ../experiments/data_aug/
python function/evaluate.py --model_dir ../experiments/data_aug/


# hyper-tuning
python function/search_hyperparams.py 

# select baseline models
python function/search_models.py 

# synthesize results
python function/synthesize_results.py