cd /root/Deborah/uw-multinerf/
apt-get update &&  apt-get -y install sudo
apt-get install vim
apt-get install git
apt-get install -y wget
apt-get install ffmpeg libsm6 libxext6 -y
apt-get install python3-pip
# apt install python3.9
# sudo ln -s /usr/bin/python3.9 /usr/bin/python

wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh &&    /bin/bash ~/miniconda.sh -b -p /opt/conda

mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_22.11.1-1-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh




# Prepare pip.
pip install --upgrade pip
pip install https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.4.1+cuda11.cudnn82-cp39-cp39-manylinux2014_x86_64.whl
# pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -r requirements.txt
# Manually install rmbrualla's `pycolmap` (don't use pip's! It's different).
git clone https://github.com/rmbrualla/pycolmap.git ./internal/pycolmap
pip install pypfm
pip install wandb
wandb login
# Confirm that all the unit tests pass.
./scripts/run_all_unit_tests.sh
:
