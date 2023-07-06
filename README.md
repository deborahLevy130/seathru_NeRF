# SeaThru-NeRF: Neural Radiance Fields In Scattering Media, CVPR 2023

#### [project page](https://sea-thru-nerf.github.io/) | [paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Levy_SeaThru-NeRF_Neural_Radiance_Fields_in_Scattering_Media_CVPR_2023_paper.pdf)

> SeaThru-NeRF: Neural Radiance Fields In Scattering Media
> [Deborah Levy](mailto:dlrun14@gmail.com) | Amit Peleg | [Naama Pearl](https://naamapearl.github.io/) | Dan Rosenbaum | [Derya Akkaynak](https://www.deryaakkaynak.com/) | [Tali Treibitz](https://www.viseaon.haifa.ac.il/) | [Simon Korman](https://www.cs.haifa.ac.il/~skorman/)
> CVPR 2023


Our implementation is based on the paper "Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields" (CVPR 2022) and their [github repository](https://github.com/google-research/multinerf).


This implementation is written in [JAX](https://github.com/google/jax).

## Setup

```
# Clone the repo.
git clone https://github.com/deborahLevy130/seathru_NeRF.git
cd seathru_NeRF
mkdir data

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

1. Calculating poses (using [COLMAP](https://colmap.github.io/install.html)):

2. Training SeaThru-NeRF:
```
./scripts/train_llff_uw.sh
```
set `SCENE` to the image set you wish to use

3. Evaluating SeaThru-NeRF on existing images:

```
./scripts/render_llff_uw.sh
```
set `SCENE` and `EXPERIMENT_NAME` to the corresponding experiment.

4. Rendering SeaThru-NeRF Novel Views:
```
./scripts/render_llff_uw.sh
```
set `SCENE` and `EXPERIMENT_NAME` to the corresponding experiment.

Your output video should now exist in the directory `ckpt/uw/${SCENE}_${EXPERIMENT_NAME}/render/`.
You will find the underwater rendering, the restored images rendering (J) and the depth maps.
## Datasets

[Here](https://drive.google.com/uc?export=download&id=1RzojBFvBWjUUhuJb95xJPSNP3nJwZWaT) you will find the underwater scenes from the paper.
Extract the files into the data folder and train SeaThru-NeRF with those scenes:

```
./scripts/train_llff_uw.sh
```

In ```'${SCENE}'``` put the name of the scene you wish to work with. 

For more datasets formats you can refer to [multinerf](https://github.com/google-research/multinerf)

For now our NeRF works on looking forward scenes.

### Running SeaThru-NeRF on your own data

In order to run SeaThru-NeRF on your own captured images of a scene, you must first run [COLMAP](https://colmap.github.io/install.html) to calculate camera poses. After you run COLMAP, you can run [this](https://github.com/Fyusion/LLFF/blob/master/imgs2poses.py) script from LLFF to get poses_bound.npy file. 
After you run COLMAP, all you need to do to load your scene in SeaThru-NeRF is ensure it has the following format:
```
my_dataset_dir/images_wb/    <--- all input images
my_dataset_dir/sparse/0/  <--- COLMAP sparse reconstruction files (cameras, images, points)
my_dataset_dir/poses_bounds.npy
```
### How to implement SeaThru-NeRF in your own NeRF

To incorporate our NeRF into an existing NeRF framework, follow these steps:

1. Incorporate the medium's module into the MLP by referring to the architecture provided in section 4.5 of the paper titled "Implementation and Optimization." You can also refer to the code available [here](https://github.com/deborahLevy130/seathru_NeRF/blob/master/internal/models.py#L866).

2. Modify the rendering equations as outlined in the paper.

3. Integrate the accuracy loss described in the paper for the object's transmission. You can refer to our implementation available [here](https://github.com/deborahLevy130/seathru_NeRF/blob/master/internal/train_utils.py#L153). If you have an alternative loss function that encourages the weights of your rendering equations to be somehow Unimodal (or close to Dirac delta function), you may use it instead of the accuracy loss. Simply apply it to the weights of the objects.





## Citation
If you use this software package, please cite whichever constituent paper(s)
you build upon, or feel free to cite this entire codebase as:

```
@inproceedings{levy2023seathru,
  title={SeaThru-NeRF: Neural Radiance Fields in Scattering Media},
  author={Levy, Deborah and Peleg, Amit and Pearl, Naama and Rosenbaum, Dan and Akkaynak, Derya and Korman, Simon and Treibitz, Tali},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={56--65},
  year={2023}
}

@misc{multinerf2022,
      title={{MultiNeRF}: {A} {Code} {Release} for {Mip-NeRF} 360, {Ref-NeRF}, and {RawNeRF}},
      author={Ben Mildenhall and Dor Verbin and Pratul P. Srinivasan and Peter Hedman and Ricardo Martin-Brualla and Jonathan T. Barron},
      year={2022},
      url={https://github.com/google-research/multinerf},
}
```
