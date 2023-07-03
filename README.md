# SeaThru-NeRF: Neural Radiance Fields In Scattering Media

#### [project page](https://sea-thru-nerf.github.io/) | [paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Levy_SeaThru-NeRF_Neural_Radiance_Fields_in_Scattering_Media_CVPR_2023_paper.pdf)

> SeaThru-NeRF: Neural Radiance Fields In Scattering Media
> [Deborah Levy](mailto:dlrun14@gmail.com) | Amit Peleg | Naama Pearl | Dan Rosenbaum | [Derya Akkaynak](https://www.deryaakkaynak.com/) | [Tali Treibitz](https://www.viseaon.haifa.ac.il/) | [Simon Korman](https://www.cs.haifa.ac.il/~skorman/)
> CVPR, 2023


Our implementation is based on the paper "Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields" (CVPR 2022) and their [github repository](https://github.com/google-research/multinerf).


This implementation is written in [JAX](https://github.com/google/jax), and
is a fork of [mip-NeRF](https://github.com/google/mipnerf).

## Setup

```
# Clone the repo.
git clone https://github.com/deborahLevy130/seathru_NeRF.git
cd seathru_NeRF_code

# Make a conda environment.
conda create --name seathruNeRF python=3.9
conda activate seathruNeRF

# Prepare pip.
conda install pip
pip install --upgrade pip

# Install requirements.
pip install -r requirements.txt

# Manually install rmbrualla's `pycolmap` (don't use pip's! It's different).
git clone https://github.com/rmbrualla/pycolmap.git ./internal/pycolmap

# Confirm that all the unit tests pass.
./scripts/run_all_unit_tests.sh
```
You'll probably also need to update your JAX installation to support GPUs or TPUs.

## Running

Example scripts for training, evaluating, and rendering can be found in
`scripts/`. You'll need to change the paths to point to wherever the datasets
are located. [Gin](https://github.com/google/gin-config) configuration files
for our model and some ablations can be found in `configs/`.


### OOM errors

You may need to reduce the batch size (`Config.batch_size`) to avoid out of memory
errors. If you do this, but want to preserve quality, be sure to increase the number
of training iterations and decrease the learning rate by whatever scale factor you
decrease batch size by.

## Using your own data

Summary: first, calculate poses. Second, train SeaThru-NeRF. Third, render a result video from the trained NeRF model.

1. Calculating poses (using COLMAP):
```
DATA_DIR=my_dataset_dir
bash scripts/local_colmap_and_resize.sh ${DATA_DIR}
```
2. Training SeaThru-NeRF:
```
python -m train \
  --gin_configs=configs/llff_256_uw.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.checkpoint_dir = '${DATA_DIR}/checkpoints'" \
  --logtostderr
```
3. Rendering SeaThru-NeRF:
```
python -m render \
  --gin_configs=configs/llff_256_uw.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.checkpoint_dir = '${DATA_DIR}/checkpoints'" \
  --gin_bindings="Config.render_dir = '${DATA_DIR}/render'" \
  --gin_bindings="Config.render_path = True" \
  --gin_bindings="Config.render_path_frames = 240" \
  --gin_bindings="Config.render_video_fps = 15" \
  --logtostderr
```
Your output video should now exist in the directory `my_dataset_dir/render/`.

## Datasets

[Here](https://drive.google.com/uc?export=download&id=1RzojBFvBWjUUhuJb95xJPSNP3nJwZWaT) you will find the underwater scenes from the paper.
Extract the files into the data folder and you can train SeaThru-NeRF with those scenes:
```
python -m train \
  --gin_configs=configs/llff_256_uw.gin \
  --gin_bindings="Config.data_dir = 'data/${DATA_DIR}'" \
  --gin_bindings="Config.checkpoint_dir = '${DATA_DIR}/checkpoints'" \
  --logtostderr
```

In ```'${DATA_DIR}'``` put the name of the scene you wish to work with. 

### Running SeaThru-NeRF on your own data

In order to run SeaThru-NeRF on your own captured images of a scene, you must first run [COLMAP](https://colmap.github.io/install.html) to calculate camera poses. You can do this using our provided script `scripts/local_colmap_and_resize.sh`. Just make a directory `my_dataset_dir/` and copy your input images into a folder `my_dataset_dir/images/`, then run:

```
After you run COLMAP, all you need to do to load your scene in SeaThru-NeRF is ensure it has the following format:
```
my_dataset_dir/images_wb/    <--- all input images
my_dataset_dir/sparse/0/  <--- COLMAP sparse reconstruction files (cameras, images, points)
```




## Citation
If you use this software package, please cite whichever constituent paper(s)
you build upon, or feel free to cite this entire codebase as:

```
@misc{multinerf2022,
      title={{MultiNeRF}: {A} {Code} {Release} for {Mip-NeRF} 360, {Ref-NeRF}, and {RawNeRF}},
      author={Ben Mildenhall and Dor Verbin and Pratul P. Srinivasan and Peter Hedman and Ricardo Martin-Brualla and Jonathan T. Barron},
      year={2022},
      url={https://github.com/google-research/multinerf},
}
```
