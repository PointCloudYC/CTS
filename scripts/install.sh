## uv environment setup rather than conda
uv venv -p 3.10
source .venv/bin/activate
uv pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install scikit-learn matplotlib pandas seaborn pyyaml opencv-python tqdm tabulate

## conda environment setup
conda create -n CTS python=3.10 -y
conda activate CTS
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install scikit-learn matplotlib pandas seaborn pyyaml opencv tqdm tabulate 
