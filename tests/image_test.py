# Copyright 2022 Google LLC
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

"""Unit tests for image."""

from absl.testing import absltest
from internal import image
import jax
from jax import random
import jax.numpy as jnp
import numpy as np


def matmul(a, b):
  """jnp.matmul defaults to bfloat16, but this helper function doesn't."""
  return jnp.matmul(a, b, precision=jax.lax.Precision.HIGHEST)


class ImageTest(absltest.TestCase):

  def test_color_correction(self):
    """Test that color correction can undo a CCM + quadratic warp + shift."""
    im_shape = (128, 128, 3)
    rng = random.PRNGKey(0)
    for _ in range(10):
      # Construct a random image.
      key, rng = random.split(rng)
      im0 = random.uniform(key, shape=im_shape, minval=0.1, maxval=0.9)

      # Construct a random linear + quadratic color transformation.
      key, rng = random.split(rng)
      ccm_scale = random.normal(key) / 10
      key, rng = random.split(rng)
      shift = random.normal(key) / 10
      key, rng = random.split(rng)
      sq_mult = random.normal(key) / 10
      key, rng = random.split(rng)
      ccm = jnp.eye(3) + random.normal(key, shape=(3, 3)) * ccm_scale

      # Apply that random transformation to the image.
      im1 = jnp.clip(
          (matmul(jnp.reshape(im0, [-1, 3]), ccm)).reshape(im0.shape) +
          sq_mult * im0**2 + shift, 0, 1)

      # Check that color correction recovers the randomly transformed image.
      im0_cc = image.color_correct(im0, im1)
      np.testing.assert_allclose(im0_cc, im1, atol=1E-5, rtol=1E-5)

  def test_psnr_mse_round_trip(self):
    """PSNR -> MSE -> PSNR is a no-op."""
    for psnr in [10., 20., 30.]:
      np.testing.assert_allclose(
          image.mse_to_psnr(image.psnr_to_mse(psnr)),
          psnr,
          atol=1E-5,
          rtol=1E-5)

  def test_ssim_dssim_round_trip(self):
    """SSIM -> DSSIM -> SSIM is a no-op."""
    for ssim in [-0.9, 0, 0.9]:
      np.testing.assert_allclose(
          image.dssim_to_ssim(image.ssim_to_dssim(ssim)),
          ssim,
          atol=1E-5,
          rtol=1E-5)

  def test_srgb_linearize(self):
    x = jnp.linspace(-1, 3, 10000)  # Nobody should call this <0 but it works.
    # Check that the round-trip transformation is a no-op.
    np.testing.assert_allclose(
        image.linear_to_srgb(image.srgb_to_linear(x)), x, atol=1E-5, rtol=1E-5)
    np.testing.assert_allclose(
        image.srgb_to_linear(image.linear_to_srgb(x)), x, atol=1E-5, rtol=1E-5)
    # Check that gradients are finite.
    self.assertTrue(
        jnp.all(jnp.isfinite(jax.vmap(jax.grad(image.linear_to_srgb))(x))))
    self.assertTrue(
        jnp.all(jnp.isfinite(jax.vmap(jax.grad(image.srgb_to_linear))(x))))

  def test_srgb_to_linear_golden(self):
    """A lazy golden test for srgb_to_linear."""
    srgb = jnp.linspace(0, 1, 64)
    linear = image.srgb_to_linear(srgb)
    linear_gt = jnp.array([
        0.00000000, 0.00122856, 0.00245712, 0.00372513, 0.00526076, 0.00711347,
        0.00929964, 0.01183453, 0.01473243, 0.01800687, 0.02167065, 0.02573599,
        0.03021459, 0.03511761, 0.04045585, 0.04623971, 0.05247922, 0.05918410,
        0.06636375, 0.07402734, 0.08218378, 0.09084171, 0.10000957, 0.10969563,
        0.11990791, 0.13065430, 0.14194246, 0.15377994, 0.16617411, 0.17913227,
        0.19266140, 0.20676863, 0.22146071, 0.23674440, 0.25262633, 0.26911288,
        0.28621066, 0.30392596, 0.32226467, 0.34123330, 0.36083785, 0.38108405,
        0.40197787, 0.42352500, 0.44573134, 0.46860245, 0.49214387, 0.51636110,
        0.54125960, 0.56684470, 0.59312177, 0.62009590, 0.64777250, 0.67615650,
        0.70525320, 0.73506740, 0.76560410, 0.79686830, 0.82886493, 0.86159873,
        0.89507430, 0.92929670, 0.96427040, 1.00000000
    ])
    np.testing.assert_allclose(linear, linear_gt, atol=1E-5, rtol=1E-5)

  def test_mse_to_psnr_golden(self):
    """A lazy golden test for mse_to_psnr."""
    mse = jnp.exp(jnp.linspace(-10, 0, 64))
    psnr = image.mse_to_psnr(mse)
    psnr_gt = jnp.array([
        43.429447, 42.740090, 42.050735, 41.361378, 40.6720240, 39.982666,
        39.293310, 38.603954, 37.914597, 37.225240, 36.5358850, 35.846527,
        35.157170, 34.467810, 33.778458, 33.089100, 32.3997460, 31.710388,
        31.021034, 30.331675, 29.642320, 28.952961, 28.2636070, 27.574250,
        26.884893, 26.195538, 25.506180, 24.816826, 24.1274700, 23.438112,
        22.748756, 22.059400, 21.370045, 20.680689, 19.9913310, 19.301975,
        18.612620, 17.923262, 17.233906, 16.544550, 15.8551940, 15.165837,
        14.4764805, 13.787125, 13.097769, 12.408413, 11.719056, 11.029700,
        10.3403420, 9.6509850, 8.9616290, 8.2722720, 7.5829163, 6.8935600,
        6.2042036, 5.5148473, 4.825491, 4.136135, 3.4467785, 2.7574227,
        2.0680661, 1.37871, 0.68935364, 0.
    ])
    np.testing.assert_allclose(psnr, psnr_gt, atol=1E-5, rtol=1E-5)


if __name__ == '__main__':
  absltest.main()
