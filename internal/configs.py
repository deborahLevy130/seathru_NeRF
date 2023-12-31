# This file was modified by Deborah Levy
#Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for handling configurations."""

import dataclasses
from typing import Any, Callable, Optional, Tuple, List

from absl import flags
from flax.core import FrozenDict
import gin
from internal import utils
import jax
import jax.numpy as jnp

# gin.add_config_file_search_path('experimental/users/barron/mipnerf360/')

configurables = {
    'jnp': [jnp.reciprocal, jnp.log, jnp.log1p, jnp.exp, jnp.sqrt, jnp.square],
    'jax.nn': [jax.nn.relu, jax.nn.softplus, jax.nn.silu],
    'jax.nn.initializers.he_normal': [jax.nn.initializers.he_normal()],
    'jax.nn.initializers.he_uniform': [jax.nn.initializers.he_uniform()],
    'jax.nn.initializers.glorot_normal': [jax.nn.initializers.glorot_normal()],
    'jax.nn.initializers.glorot_uniform': [
        jax.nn.initializers.glorot_uniform()
    ],
}

for module, configurables in configurables.items():
    for configurable in configurables:
        gin.config.external_configurable(configurable, module=module)


@gin.configurable()
@dataclasses.dataclass
class Config:
    """Configuration flags for everything."""

    dataset_loader: str = 'llff'  # The type of dataset loader to use.
    batching: str = 'all_images'  # Batch composition, [single_image, all_images].
    batch_size: int = 16384  # The number of rays/pixels in each batch.
    patch_size: int = 1  # Resolution of patches sampled for training batches.
    factor: int = 0  # The downsample factor of images, 0 for no downsampling.
    load_alphabetical: bool = True  # Load images in COLMAP vs alphabetical
    # ordering (affects heldout test set).
    forward_facing: bool = True  # Set to True for forward-facing LLFF captures.
    render_path: bool = False  # If True, render a path. Used only by LLFF.
    llffhold: int = 8  # Use every Nth image for the test set. Used only by LLFF.
    # If true, use all input images for training.
    llff_use_all_images_for_training: bool = False
    use_tiffs: bool = False  # If True, use 32-bit TIFFs. Used only by Blender.
    compute_disp_metrics: bool = False
    compute_normal_metrics: bool = False
    gc_every: int = 10000  # The number of steps between garbage collections.
    disable_multiscale_loss: bool = False  # If True, disable multiscale loss.
    randomized: bool = True  # Use randomized stratified sampling.
    near: float = 2.  # Near plane distance.
    far: float = 6.  # Far plane distance.
    checkpoint_dir: Optional[str] = None  # Where to log checkpoints.
    render_dir: Optional[str] = None  # Output rendering directory.
    data_dir: Optional[str] = None  # Input data directory.
    vocab_tree_path: Optional[str] = None  # Path to vocab tree for COLMAP.
    render_chunk_size: int = 16384  # Chunk size for whole-image renderings.
    num_showcase_images: int = 5  # The number of test-set images to showcase.
    deterministic_showcase: bool = True  # If True, showcase the same images.
    vis_num_rays: int = 8  # The number of rays to visualize.
    # Decimate images for tensorboard (ie, x[::d, ::d]) to conserve memory usage.
    vis_decimate: int = 0

    # Only used by train.py:
    max_steps: int = 250000  # The number of optimization steps.
    early_exit_steps: Optional[int] = None  # Early stopping, for debugging.
    checkpoint_every: int = 25000  # The number of steps to save a checkpoint.
    print_every: int = 100  # The number of steps between reports to tensorboard.
    train_render_every: int = 5000  # Steps between test set renders when training
    cast_rays_in_train_step: bool = False  # If True, compute rays in train step.
    data_loss_type: str = 'rawnerf'  # What kind of loss to use ('mse' or 'charb').
    charb_padding: float = 0.001  # The padding used for Charbonnier loss.
    data_loss_mult: float = 1.0  # Mult for the finest data term in the loss.
    data_coarse_loss_mult: float = 0.  # Multiplier for the coarser data terms.
    interlevel_loss_mult: float = 1.0  # Mult. for the loss on the proposal MLP.

    weight_decay_mults: FrozenDict[str, Any] = FrozenDict({})  # Weight decays.
    # An example that regularizes the NeRF and the first layer of the prop MLP:
    #   weight_decay_mults = {
    #       'NerfMLP_0': 0.00001,
    #       'PropMLP_0/Dense_0': 0.001,
    #   }
    # Any model parameter that isn't specified gets a mult of 0. See the
    # train_weight_l2_* parameters in TensorBoard to know what can be regularized.

    lr_init: float = 0.002  # The initial learning rate.
    lr_final: float = 0.00002  # The final learning rate.
    lr_delay_steps: int = 512  # The number of "warmup" learning steps.
    lr_delay_mult: float = 0.01  # How much sever the "warmup" should be.
    adam_beta1: float = 0.9  # Adam's beta2 hyperparameter.
    adam_beta2: float = 0.999  # Adam's beta2 hyperparameter.
    adam_eps: float = 1e-6  # Adam's epsilon hyperparameter.
    grad_max_norm: float = 0.001  # Gradient clipping magnitude, disabled if == 0.
    grad_max_val: float = 0.  # Gradient clipping value, disabled if == 0.
    distortion_loss_mult: float = 0.01  # Multiplier on the distortion loss.

    # Only used by eval.py:
    eval_only_once: bool = True  # If True evaluate the model only once, ow loop.
    eval_save_output: bool = True  # If True save predicted images to disk.
    eval_save_ray_data: bool = False  # If True save individual ray traces.
    eval_render_interval: int = 1  # The interval between images saved to disk.
    eval_dataset_limit: int = jnp.iinfo(jnp.int32).max  # Num test images to eval.
    eval_quantize_metrics: bool = True  # If True, run metrics on 8-bit images.
    eval_crop_borders: int = 0  # Ignore c border pixels in eval (x[c:-c, c:-c]).
    eval_on_train: bool = False

    # Only used by render.py
    render_video_fps: int = 60  # Framerate in frames-per-second.
    render_video_crf: int = 18  # Constant rate factor for ffmpeg video quality.
    render_path_frames: int = 120  # Number of frames in render path.
    z_variation: float = 0.  # How much height variation in render path.
    z_phase: float = 0.  # Phase offset for height variation in render path.
    render_dist_percentile: float = 0.5  # How much to trim from near/far planes.
    render_dist_curve_fn: Callable[..., Any] = jnp.log  # How depth is curved.
    render_path_file: Optional[str] = None  # Numpy render pose file to load.
    render_job_id: int = 0  # Render job id.
    render_num_jobs: int = 1  # Total number of render jobs.
    render_resolution: Optional[Tuple[int, int]] = None  # Render resolution, as
    # (width, height).
    render_focal: Optional[float] = None  # Render focal length.
    render_camtype: Optional[str] = None  # 'perspective', 'fisheye', or 'pano'.
    render_spherical: bool = False  # Render spherical 360 panoramas.
    render_save_async: bool = True  # Save to CNS using a separate thread.

    render_spline_keyframes: Optional[str] = None  # Text file containing names of
    # images to be used as spline
    # keyframes, OR directory
    # containing those images.
    render_spline_n_interp: int = 30  # Num. frames to interpolate per keyframe.
    render_spline_degree: int = 5  # Polynomial degree of B-spline interpolation.
    render_spline_smoothness: float = .03  # B-spline smoothing factor, 0 for
    # exact interpolation of keyframes.
    # Interpolate per-frame exposure value from spline keyframes.
    render_spline_interpolate_exposure: bool = False

    # # Flags for raw datasets.
    rawnerf_mode: bool = False  # Load raw images and train in raw color space.
    exposure_percentile: float = 97.  # Image percentile to expose as white.
    num_border_pixels_to_mask: int = 0  # During training, discard N-pixel border
    # around each input image.
    apply_bayer_mask: bool = False  # During training, apply Bayer mosaic mask.
    autoexpose_renders: bool = False  # During rendering, autoexpose each image.
    # For raw test scenes, use affine raw-space color correction.
    eval_raw_affine_cc: bool = False

    # underwater flags
    use_uw_mlp: bool = False
    use_uw_acc_weights_loss: bool = False  # use accuracy loss (equation (27) from the paper) on object's weights sum
    use_uw_acc_trans_loss: bool = False  # use accuracy loss (equation (27) from the paper) on object's transmission like the paper
    use_uw_sig_med_loss: bool = False  # use std loss on medium's densities to imply smoothness on the densities - not in the paper
    uw_initial_acc_weights_loss_mult: float = 0.01  # initial mult factor for acc_weights_loss can be changed after uw_decay_acc iterations
    uw_initial_acc_trans_loss_mult: float = 0.01  # initial mult factor for acc_trans_loss can be changed after uw_decay_acc iterations
    uw_final_acc_weights_loss_mult: float = 0.01  # final mult factor for acc_weights_loss  after uw_decay_acc iterations
    uw_final_acc_trans_loss_mult: float = 0.01  # final mult factor for acc_trans_loss after uw_decay_acc iterations
    uw_acc_loss_factor: float = 6  # factor which encourages one of the terms in the accuracy loss to be more dominant - for the trans it encourages it to be 1 and for the weights zero.
    uw_sig_med_mult: float = 0.01  # mult factor for std loss - not in the paper!!
    # uw_usual_rendering : int = 600
    # uw_final_usual_rendering: int = 1500
    uw_old_model: bool = False  # If True same sigmas for attenuation and backscatter.
    uw_decay_acc: int = 5000  # number of iterations to change acc_loss_mult value
    gen_eq: bool = False  # use equations (11)-(14) from the paper instead of (22)
    uw_atten_xyz: bool = False  # If True use rgb xyz coordinates also as input for sigma_atten prediction
    uw_fog_model: bool = False  # If True same sigmas for attenuation and backscatter and for the same for all color channels.
    uw_rgb_dir: bool = False #If True use view_dir also as input for rgb_obj

    extra_samples: bool = False # If true add extra samples to the beginning of the bs component - (for fog sim)


def define_common_flags():
    # Define the flags used by both train.py and eval.py
    flags.DEFINE_string('mode', None, 'Required by GINXM, not used.')
    flags.DEFINE_string('base_folder', None, 'Required by GINXM, not used.')
    flags.DEFINE_multi_string('gin_bindings', None, 'Gin parameter bindings.')
    flags.DEFINE_multi_string('gin_configs', None, 'Gin config files.')


def load_config(save_config=True):
    """Load the config, and optionally checkpoint it."""
    gin.parse_config_files_and_bindings(
        flags.FLAGS.gin_configs, flags.FLAGS.gin_bindings, skip_unknown=True)
    config = Config()
    if save_config and jax.host_id() == 0:
        utils.makedirs(config.checkpoint_dir)
        with utils.open_file(config.checkpoint_dir + '/config.gin', 'w') as f:
            f.write(gin.config_str())
    return config
