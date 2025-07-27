# Classification of Chinese Traditional Architectural Styles

This project implements and evaluates deep learning models for classifying Chinese traditional architectural styles. It provides a framework for training, evaluating, and interpreting various CNN models like AlexNet, ResNet, and DenseNet.

Paper: [Towards Classification of Architectural Styles of Traditional Settlements using DL: A Dataset, a New Framework, and Its Interpretability](https://www.mdpi.com/2072-4292/14/20/5250), Qing HAN, Chao YIN*, etc., Remote Sensing, 2022

## Features

- **Model Training**: Train baseline models and transfer-learning models with data augmentation(e.g, conventional data augmentation, learning-based data augmentation(AutoAugment)).
- **Comprehensive Evaluation**: Calculate metrics including accuracy, precision, recall, and F1-score.
- **Confusion Matrix**: Generate and save confusion matrices to visualize model performance on the test set.
- **Model Interpretability**: Generate Class Activation Maps (CAMs) to understand which parts of an image a model focuses on for its predictions.
- **Results Synthesis**: Aggregate results from multiple experiments into a consolidated report.

## Getting Started

Follow these steps to set up the project environment and run the experiments.

### Prerequisites

- Python 3.8+
- Git
- We recommend using a virtual environment (`uv` or `conda` or`venv`).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/PointCloudYC/CTS.git
    cd CTS
    ```

2.  **Create a virtual environment and activate it:**
    ```bash
    # use uv
    uv venv -p 3.10
    source .venv/bin/activate

    # or use conda
    conda create -n CTS python=3.10 -y
    conda activate CTS

    # or use venv   
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    # use uv
    uv pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
    uv pip install scikit-learn matplotlib pandas seaborn pyyaml opencv-python tqdm tabulate

    # or use conda
    conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
    pip install scikit-learn matplotlib pandas seaborn pyyaml opencv-python tqdm tabulate 

    # detailed packatges are in requirements.txt
    ```

### Dataset

1.  Download the dataset from the [ArchiStyle dataset](https://hkustconnect-my.sharepoint.com/:u:/g/personal/cyinac_connect_ust_hk/EW357p4zW0JKoadv5Ywcp7oBFBZ63RKSpjeRBXFokeIm-A?e=AQiD5u).
2.  Unzip the contents into the `data/` directory.

The expected directory structure is(two dataset variants, the paper use the `ArchiStyle-v1`):

```
CTS/
├── data/
│   └── ArchiStyle-v1/
│       └── train/
│       └── val/
│       └── test/
│   └── ArchiStyle-v2/
│       └── train/
│       └── val/
│       └── test/
├── function/
├── model/
└── ...
```

## Usage

This project uses shell scripts to streamline running experiments. The results of each experiment, including logs, model checkpoints, and generated images, are saved in the `experiments-v1/` for `ArchiStyle-v1` and `experiments-v2/` for `ArchiStyle-v2` directories.

### Running Pre-configured Experiments

-   **Run baseline models on ArchiStyle-v1 for the paper:**
    ```bash
    bash scripts/run-v1.sh
    ```
-   **Run transfer-learning models on ArchiStyle-v2:**
    ```bash
    bash scripts/run-v2.sh
    ```

### Generating a Confusion Matrix

To evaluate a trained model and generate a confusion matrix, use the `evaluate.py` script with the `--save_confusion_matrix` flag. The matrix will be saved in the experiment's `images/` directory.

```bash
python function/evaluate.py \
    --model_dir experiments-v1/alexnet_pretrained \
    --data_dir data/ArchiStyle-v1 \
    --model_name alexnet \
    --pretrained \
    --save_confusion_matrix
```

### Generating Class Activation Maps (CAMs)

To generate CAMs for model interpretability, use the `generate_cam.py` script. The CAM images will be saved in the experiment's `images/cams/` directory.

```bash
python function/generate_cam.py \
    --model_dir experiments-v1/alexnet_pretrained \
    --data_dir data/ArchiStyle-v1 \
    --model_name alexnet \
    --pretrained \
    --num_images_per_class 5
```

### Synthesizing Results

After running experiments, you can generate a CSV report summarizing the metrics from all models within an experiment directory:

```bash
# For baseline models
python function/synthesize_results.py --parent_dir experiments-v1

# For transfer-learning models
python function/synthesize_results.py --parent_dir experiments-v2
```

## Project Structure

```
├── data/                 # Datasets
├── experiments-v1/       # Results for baseline and pre-trained models on ArchiStyle-v1
├── experiments-v2/       # Results for baseline and pre-trained models on ArchiStyle-v2
├── function/             # Core scripts (train, evaluate, etc.)
├── grad-cam/             # Original Grad-CAM implementation
├── model/                # Model definitions (net.py) and data loaders
├── scripts/              # Shell scripts to run experiments
└── ...
```

## Citation

If you use this project in your research, please cite the original paper:

```
@article{CTS,
    Author = {Qing HAN, Chao YIN*, Yunyuan DENG, Peilin LIU},
    Title = {Towards Classification of Architectural Styles of Chinese Traditional Settlements using Deep Learning: A Dataset, a New Framework, and Its Interpretability},
    Journal = {Remote Sensing},
    Year = {2022}
}
```