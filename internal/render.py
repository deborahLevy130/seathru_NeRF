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

"""Helper functions for shooting and rendering rays."""

from internal import stepfun
import jax.numpy as jnp
import jax


def lift_gaussian(d, t_mean, t_var, r_var, diag):
    """Lift a Gaussian defined along a ray to 3D coordinates."""
    mean = d[..., None, :] * t_mean[..., None]

    d_mag_sq = jnp.maximum(1e-10, jnp.sum(d ** 2, axis=-1, keepdims=True))

    if diag:
        d_outer_diag = d ** 2
        null_outer_diag = 1 - d_outer_diag / d_mag_sq
        t_cov_diag = t_var[..., None] * d_outer_diag[..., None, :]
        xy_cov_diag = r_var[..., None] * null_outer_diag[..., None, :]
        cov_diag = t_cov_diag + xy_cov_diag
        return mean, cov_diag
    else:
        d_outer = d[..., :, None] * d[..., None, :]
        eye = jnp.eye(d.shape[-1])
        null_outer = eye - d[..., :, None] * (d / d_mag_sq)[..., None, :]
        t_cov = t_var[..., None, None] * d_outer[..., None, :, :]
        xy_cov = r_var[..., None, None] * null_outer[..., None, :, :]
        cov = t_cov + xy_cov
        return mean, cov


def conical_frustum_to_gaussian(d, t0, t1, base_radius, diag, stable=True):
    """Approximate a conical frustum as a Gaussian distribution (mean+cov).

    Assumes the ray is originating from the origin, and base_radius is the
    radius at dist=1. Doesn't assume `d` is normalized.

    Args:
      d: jnp.float32 3-vector, the axis of the cone
      t0: float, the starting distance of the frustum.
      t1: float, the ending distance of the frustum.
      base_radius: float, the scale of the radius as a function of distance.
      diag: boolean, whether or the Gaussian will be diagonal or full-covariance.
      stable: boolean, whether or not to use the stable computation described in
        the paper (setting this to False will cause catastrophic failure).

    Returns:
      a Gaussian (mean and covariance).
    """
    if stable:
        # Equation 7 in the paper (https://arxiv.org/abs/2103.13415).
        mu = (t0 + t1) / 2  # The average of the two `t` values.
        hw = (t1 - t0) / 2  # The half-width of the two `t` values.
        eps = jnp.finfo(jnp.float32).eps
        t_mean = mu + (2 * mu * hw ** 2) / jnp.maximum(eps, 3 * mu ** 2 + hw ** 2)
        denom = jnp.maximum(eps, 3 * mu ** 2 + hw ** 2)
        t_var = (hw ** 2) / 3 - (4 / 15) * hw ** 4 * (12 * mu ** 2 - hw ** 2) / denom ** 2
        r_var = (mu ** 2) / 4 + (5 / 12) * hw ** 2 - (4 / 15) * (hw ** 4) / denom
    else:
        # Equations 37-39 in the paper.
        t_mean = (3 * (t1 ** 4 - t0 ** 4)) / (4 * (t1 ** 3 - t0 ** 3))
        r_var = 3 / 20 * (t1 ** 5 - t0 ** 5) / (t1 ** 3 - t0 ** 3)
        t_mosq = 3 / 5 * (t1 ** 5 - t0 ** 5) / (t1 ** 3 - t0 ** 3)
        t_var = t_mosq - t_mean ** 2
    r_var *= base_radius ** 2
    return lift_gaussian(d, t_mean, t_var, r_var, diag)


def cylinder_to_gaussian(d, t0, t1, radius, diag):
    """Approximate a cylinder as a Gaussian distribution (mean+cov).

    Assumes the ray is originating from the origin, and radius is the
    radius. Does not renormalize `d`.

    Args:
      d: jnp.float32 3-vector, the axis of the cylinder
      t0: float, the starting distance of the cylinder.
      t1: float, the ending distance of the cylinder.
      radius: float, the radius of the cylinder
      diag: boolean, whether or the Gaussian will be diagonal or full-covariance.

    Returns:
      a Gaussian (mean and covariance).
    """
    t_mean = (t0 + t1) / 2
    r_var = radius ** 2 / 4
    t_var = (t1 - t0) ** 2 / 12
    return lift_gaussian(d, t_mean, t_var, r_var, diag)


def cast_rays(tdist, origins, directions, radii, ray_shape, diag=True):
    """Cast rays (cone- or cylinder-shaped) and featurize sections of it.

    Args:
      tdist: float array, the "fencepost" distances along the ray.
      origins: float array, the ray origin coordinates.
      directions: float array, the ray direction vectors.
      radii: float array, the radii (base radii for cones) of the rays.
      ray_shape: string, the shape of the ray, must be 'cone' or 'cylinder'.
      diag: boolean, whether or not the covariance matrices should be diagonal.

    Returns:
      a tuple of arrays of means and covariances.
    """
    t0 = tdist[..., :-1]
    t1 = tdist[..., 1:]
    if ray_shape == 'cone':
        gaussian_fn = conical_frustum_to_gaussian
    elif ray_shape == 'cylinder':
        gaussian_fn = cylinder_to_gaussian
    else:
        raise ValueError('ray_shape must be \'cone\' or \'cylinder\'')
    means, covs = gaussian_fn(directions, t0, t1, radii, diag)
    means = means + origins[..., None, :]
    return means, covs


def get_sorted_indices(arr1, arr2):

    batch_size = arr1.shape[0]


    # Create an empty list to store the sorted indexes
    sorted_indexes1 = []
    sorted_indexes2 = []

    # Loop through the arrays in the batch
    for (array1,array2) in zip(arr1,arr2):
        # Get the indexes of the sorted array

     # Combine the two arrays and sort them
        combined = jnp.concatenate((array1, array2),axis=-1)
        combined = jnp.sort(combined)
        # Get the indices of each element in the sorted array
        indices1  = jnp.where(jnp.isin( array1,combined),size=array1.size)[0]
        indices2  = jnp.where(jnp.isin( array2,combined),size=array2.size)[0]
        sorted_indexes1.append(indices1)
        sorted_indexes2.append(indices2)
    if batch_size>1:
        return jnp.expand_dims(jnp.array(sorted_indexes1),axis=[1,2]), jnp.expand_dims(jnp.array(sorted_indexes2),axis=[1,2])
    else:
        return jnp.array(sorted_indexes1), jnp.array(sorted_indexes2)




def compute_alpha_weights_uw(density_obj, sigma_bs, sigma_atten, tdist, dirs,c_med, opaque_background=False):
    """Helper function for computing alpha compositing weights."""
    # tdist =  jnp.concatenate([
    #   jnp.zeros_like(tdist[..., -1:]),
    #   tdist[..., :-1]
    # ],
    #   axis=-1)
    t_delta = tdist[..., 1:] - tdist[..., :-1]
    # t_dist_bs = (jnp.linspace(0, jax.lax.stop_gradient(tdist[..., 0]), 33 ,axis=-1)) #TODO: uncomment for extra samples
    # t_dist_bs_sort = (jnp.concatenate([ t_dist_bs[...,:-1], jax.lax.stop_gradient(tdist)], axis=-1))# TODO: uncomment for extra samples
    # t_delta_bs = t_dist_bs_sort[..., 1:] - t_dist_bs_sort[..., :-1] #TODO: uncomment for extra samples
    # # # # indices_t_dist, indices2 = get_sorted_indices(tdist, t_dist_bs[...,:-1])
    # trans_obj_bs = jnp.ones_like(t_dist_bs) #TODO: uncomment for extra samples
    # trans_obj_bs = trans_obj_bs[...,:-1] #TODO: uncomment for extra samples
    # t_delta_bs = jnp.repeat(jnp.expand_dims(t_delta_bs, -1), t_delta.shape[-1], 1)
    # t_delta = rendering['tdist'][..., 1:] - rendering['tdist'][..., :-1]
    # jax.debug.print("{y}", y=t_delta[:, 0])
    # delta = jnp.concatenate([
    #   t_delta[..., :-1],
    #   density_obj[..., -1:] *
    #   jnp.zeros_like(t_delta[..., -1:], jnp.inf)
    # ],
    #   axis=-1)
    delta = t_delta * jnp.linalg.norm(dirs[..., None, :], axis=-1)
    delta_bs = jax.lax.stop_gradient(t_delta) * jnp.linalg.norm(dirs[..., None, :], axis=-1) #TODO: uncomment for extra samples
    # delta_bs = jax.lax.stop_gradient(t_delta) * jnp.linalg.norm(dirs[..., None, :], axis=-1)
        # delta_bs = t_delta* jnp.linalg.norm(dirs[..., None, :], axis=-1)






    # density_delta = jnp.where(density_obj<=delta, 0, density_obj * delta)
    # delta_bs = delta
    density_delta = density_obj * delta


    # delta_density = density_obj*delta_bs
    # density_delta = jnp.concatenate([
    #       density_delta[..., :-1],
    #       density_obj[..., -1:] *
    #       jnp.full_like(density_delta[..., -1:],jnp.inf)
    #   ],
    #                                   axis=-1)

    # Equivalent to making the final t-interval infinitely wide.
    # bs_delta = sigma_bs[..., None, :] * jax.lax.stop_gradient
    bs_delta = sigma_bs[..., None, :] * (delta_bs[..., None])

    # bs_delta_alpha = sigma_bs[..., None, :] * (delta_bs[..., None])
    # bs_delta_trans = sigma_atten[..., None, :] * (delta_bs[..., None])

    # bs_delta_uni = sigma_bs[..., None, :] * delta_bs_uni[...,None]
    # bs_delta = jnp.concatenate([
    #       bs_delta[..., :-1,:],
    #       jnp.full_like(bs_delta[..., -1:,:], jnp.inf)
    #   ],
    #                                   axis=-2)

    alpha_bs = 1 - jnp.exp(-bs_delta)
    # alpha_bs = 1-jnp.exp(-jnp.concatenate([
    #     jnp.zeros_like(bs_delta[..., :1,:]),
    #     bs_delta[..., :-1,:]
    # ],
    #                                  axis=-2))

    trans_bs = jnp.exp(-jnp.concatenate([
        jnp.zeros_like(bs_delta[..., :1, :]),
        jnp.cumsum(bs_delta[..., :-1, :], axis=-2)
    ],
        axis=-2))
    # trans_bs = jnp.exp(
    #     -jnp.cumsum(bs_delta, axis=-2)
    #
    #                                  )
    # delta_bs_atten = delta_bs[...,:(t_dist_bs.shape[-1]-1)]
    # print(delta_bs_atten[..., None].shape)
    atten_delta = sigma_atten[..., None, :] * (delta[..., None])
    # atten_delta_bs = sigma_atten[..., None, :] *delta_bs_atten[..., None]# TODO: uncomment for extra samples
    # atten_delta = sigma_bs[..., None, :] * (delta[..., None])
    # atten_delta_bs = sigma_bs[..., None, :] * delta_bs_atten[..., None]
    # atten_delta = sigma_atten* delta[...,None]

    # atten_delta = jnp.concatenate([
    #       atten_delta[..., :-1,:],
    #       jnp.full_like(atten_delta[..., -1:,:], jnp.inf)
    #   ],
    #                                   axis=-2)
    # print(atten_delta_bs.sum(axis=-2)[..., None, :].shape)
    # trans_atten = jnp.exp(-jnp.concatenate([
    #     jnp.zeros_like(atten_delta[..., :1, :]),
    #     jnp.cumsum(atten_delta[..., :-1, :], axis=-2)+atten_delta_bs.sum(axis=-2)[..., None, :] ],
    #     axis=-2))
    trans_atten = jnp.exp(-jnp.concatenate([
        jnp.zeros_like(atten_delta[..., :1, :]),
        jnp.cumsum(atten_delta[..., :-1, :], axis=-2)
    ],
        axis=-2))

    # trans_atten = jnp.exp(-
    #     jnp.cumsum(atten_delta, axis=-2))

    alpha = 1 - jnp.exp(-density_delta)
    trans = jnp.exp(-jnp.concatenate([
        jnp.zeros_like(density_delta[..., :1]),
        jnp.cumsum(density_delta[..., :-1], axis=-1)
    ],
        axis=-1))

    # trans_obj_clipped = jnp.where(
    #    trans>= 0.5,
    #     1, 0)
    #
    # trans_obj_bs.at[...,
    #                 (trans_obj_bs.shape[-1]-t_delta.shape[-1]):].set(trans)
    # (`weights` is only guaranteed to sum to <= 1, not == 1).

    # trans_obj_bs = jnp.concatenate([trans_obj_bs,trans],axis=-1) #TODO: uncomment for extra samples
    # for ind in range(trans_obj_bs.shape[-1]-1)[::-1]:
    #     if ~jnp.isin(jnp.array(ind)[...,None],indices_t_dist):
    #         trans_obj_bs.at[...,ind].set(trans_obj_bs[...,ind + 1][0])

    # trans = jnp.exp(-
    #     jnp.cumsum(density_delta, axis=-1))

    # trans_o_bs = jnp.concatenate([trans[...,:1],trans[...,1:]*jnp.exp(-density_delta[...,1:])],axis=-1)
    weights = alpha * trans
    # weights = jax.nn.relu(weights-0.1)
    # bs_weights = (alpha_bs * trans_bs*trans_obj_bs[...,None]* c_med[..., None, :]).sum(axis=-2) #TODO: uncomment for extra samples
    bs_weights = alpha_bs * trans_bs
    #

    # density_obj_small = density_obj
    # idx = jnp.where(density_obj<4,size=density_obj.size)
    # density_obj_small=density_obj_small.at[idx[-1]].set(0)
    # alpha_aug = 1 - jnp.exp(-density_obj_small*delta)
    # trans_aug = jnp.exp(-jnp.concatenate([
    #     jnp.zeros_like(-(density_obj_small*delta)[..., :1]),
    #     jnp.cumsum(-(density_obj_small*delta)[..., :-1], axis=-1)
    # ],
    #                                  axis=-1))
    # weights_aug = alpha_aug*trans_aug
    # bs_weights = alpha_bs * trans_bs
    # return weights, (alpha_bs * trans_bs*trans_obj_bs[...,None]), trans, bs_weights, trans_atten, alpha_bs, trans_bs #TODO: uncomment for extra samples
    return weights, alpha, trans, bs_weights, trans_atten, alpha_bs, trans_bs


def compute_alpha_weights_uw_simon(density_obj, sigma_bs, sigma_atten, tdist, dirs, opaque_background=False):
    """Helper function for computing alpha compositing weights."""
    t_delta = tdist[..., 1:] - tdist[..., :-1]
    # t_dist_bs = (jnp.linspace(0, jax.lax.stop_gradient(tdist[..., 0]), 33, axis=-1))
    # t_dist_bs_sort = (jnp.concatenate([t_dist_bs[..., :-1], jax.lax.stop_gradient(tdist)], axis=-1))
    # t_delta_bs = t_dist_bs_sort[..., 1:] - t_dist_bs_sort[..., :-1]
    # # # # indices_t_dist, indices2 = get_sorted_indices(tdist, t_dist_bs[...,:-1])
    # trans_obj_bs = jnp.ones_like(t_dist_bs)
    # trans_obj_bs = trans_obj_bs[..., :-1]


    delta_bs = jax.lax.stop_gradient(t_delta) * jnp.linalg.norm(dirs[..., None, :], axis=-1)


    delta = t_delta * jnp.linalg.norm(dirs[..., None, :], axis=-1)
    density_delta = density_obj * delta
    # density_delta = jnp.concatenate([
    #       density_delta[..., :-1],
    #       density_obj[..., -1:] *
    #       jnp.full_like(density_delta[..., -1:],jnp.inf)
    #   ],
    #                                   axis=-1)

    # Equivalent to making the final t-interval infinitely wide.
    bs_delta = sigma_bs[..., None, :] * delta_bs[..., None]
    # bs_delta = jnp.concatenate([
    #       bs_delta[..., :-1,:],
    #       jnp.full_like(bs_delta[..., -1:,:], jnp.inf)
    #   ],
    #                                   axis=-2)

    alpha_bs = 1 - jnp.exp(-bs_delta) * jnp.exp(-density_delta)[..., None]
    trans_bs = jnp.exp(-jnp.concatenate([
        jnp.zeros_like(bs_delta[..., :1, :]),
        jnp.cumsum(bs_delta[..., :-1, :], axis=-2)
    ],
        axis=-2))
    # bs_weights = alpha_bs*trans_bs
    #
    atten_delta = sigma_atten[..., None, :] * delta_bs[...,None]
    alpha_atten = 1 - jnp.exp(-atten_delta) * jnp.exp(-density_delta)[..., None]
    trans_atten = jnp.exp(-jnp.concatenate([
        jnp.zeros_like(atten_delta[..., :1,:]),
        jnp.cumsum(atten_delta[..., :-1,:], axis=-2)
    ],
                                     axis=-2))

    alpha = 1 - jnp.exp(-density_delta)
    trans = jnp.exp(-jnp.concatenate([
        jnp.zeros_like(density_delta[..., :1]),
        jnp.cumsum(density_delta[..., :-1], axis=-1)
    ],
        axis=-1))
    weights = alpha * trans
    # trans_obj_bs = jnp.concatenate([trans_obj_bs,trans],axis=-1)

    return weights, alpha, trans, alpha_bs, trans_bs,alpha_atten,trans_atten


def compute_alpha_weights(density, tdist, dirs, opaque_background=False):
    """Helper function for computing alpha compositing weights."""
    t_delta = tdist[..., 1:] - tdist[..., :-1]

    delta = t_delta * jnp.linalg.norm(dirs[..., None, :], axis=-1)
    # density_delta = jnp.where(density <= 10 * delta, 0, density * delta)
    density_delta = density * delta

    if opaque_background:
        # Equivalent to making the final t-interval infinitely wide.
        density_delta = jnp.concatenate([
            density_delta[..., :-1],
            jnp.full_like(density_delta[..., -1:], jnp.inf)
        ],
            axis=-1)

    alpha = 1 - jnp.exp(-density_delta)
    trans = jnp.exp(-jnp.concatenate([
        jnp.zeros_like(density_delta[..., :1]),
        jnp.cumsum(density_delta[..., :-1], axis=-1)
    ],
        axis=-1))
    weights = alpha * trans
    return weights, alpha, trans


# TODO: add uw_volometric_rendering

def volumetric_rendering_uw(density, sigma_bs, rgbs, c_med, bs_weights,
                            trans_atten, trans, weights, alpha,
                            tdist,
                            bg_rgbs,
                            t_far,
                            compute_extras,
                            extras=None):
    """Volumetric Rendering Function.

    Args:
      rgbs: jnp.ndarray(float32), color, [batch_size, num_samples, 3]
      weights: jnp.ndarray(float32), weights, [batch_size, num_samples].
      tdist: jnp.ndarray(float32), [batch_size, num_samples].
      bg_rgbs: jnp.ndarray(float32), the color(s) to use for the background.
      t_far: jnp.ndarray(float32), [batch_size, 1], the distance of the far plane.
      compute_extras: bool, if True, compute extra quantities besides color.
      extras: dict, a set of values along rays to render by alpha compositing.

    Returns:
      rendering: a dict containing an rgb image of size [batch_size, 3], and other
        visualizations if compute_extras=True.
    """
    eps = jnp.finfo(jnp.float32).eps
    rendering = {}

    acc = weights.sum(axis=-1)
    bg_w = jnp.minimum(acc[..., None], 1 - acc[..., None])  # The weight of the background.
    # rgb = (weights[..., None] * rgbs).sum(axis=-2) + bg_w * bg_rgbs
    # sum_weights = (weights[..., None]*trans_atten +bs_weights * trans[..., None]).sum(axis=-2)

    # norm_weights_direct = jax.nn.softmax(weights,axis=-1)#*trans_atten,axis=-2)
    # norm_weights_bs = jax.nn.softmax(bs_weights * trans[..., None] ,axis=-2)

    J = (weights[..., None] * rgbs).sum(axis=-2)
    direct = (weights[..., None] * trans_atten * rgbs).sum(axis=-2) ##(bg_w * bg_rgbs).sum(axis=-2)
    t_mids = 0.5 * (tdist[..., :-1] + tdist[..., 1:])

    # bs = (c_med[..., None, :]*(1-jnp.exp(-sigma_bs[..., None, :]*weights[..., None]*t_mids[..., None]))).sum(axis=-2)
    bs = (trans[..., None]*bs_weights* c_med[..., None, :]).sum(axis=-2)  # TODO Naama changes, broadcast for c_med
    # bs = bs_weights #TODO: uncomment for extra samples
    # J = (weights[..., None] * rgbs).sum(axis=-2)
    # direct = (norm_weights_direct * rgbs).sum(axis=-2)
    # bs = (norm_weights_bs* c_med[..., None, :]).sum(axis=-2)
    rgb = direct + bs  # + bg_w * bg_rgb# J

    rendering['rgb'] = rgb
    rendering['bs'] = bs
    rendering['density'] = density
    rendering['J'] = J
    rendering['direct'] = direct
    rendering['c_med'] = c_med
    rendering['t_dist'] = tdist
    # rendering['direct_weights'] = (weights[..., None] * trans_atten).sum(axis=-2)
    # rendering['bs_weights'] = alpha.sum(axis=-2)
    # rendering['bs_weights'] = (trans[..., None]*bs_weights).sum(axis=-2)
    expectation = lambda x: (weights * x).sum(axis=-1) / jnp.maximum(eps, acc)
    rendering['distance_mean'] = (
        jnp.clip(
            jnp.nan_to_num(jnp.exp(expectation(jnp.log(t_mids))), jnp.inf),
            tdist[..., 0], tdist[..., -1]))
    t_aug = jnp.concatenate([tdist, t_far], axis=-1)
    weights_aug = jnp.concatenate([weights, bg_w], axis=-1)

    ps = [5, 50, 95]
    distance_percentiles = stepfun.weighted_percentile(t_aug, weights_aug, ps)
    for i, p in enumerate(ps):
        s = 'median' if p == 50 else 'percentile_' + str(p)
        rendering['distance_' + s] = distance_percentiles[..., i]

    if compute_extras:
        rendering['acc'] = acc

        if extras is not None:
            for k, v in extras.items():
                if v is not None:
                    rendering[k] = (weights[..., None] * v).sum(axis=-2)
        # acc2 = weights_aug.sum(axis=-1)
        # acc_2 = ((weights*jnp.mean(trans_atten,axis=-1))+(jnp.mean(bs_weights,axis=-1) * trans)).sum(axis=-1)
        # expectation = lambda x: (((weights*jnp.mean(trans_atten,axis=-1))+(jnp.mean(bs_weights,axis=-1) * trans))*x).sum(axis=-1) / jnp.maximum(eps, acc_2)
        expectation = lambda x: (weights * x).sum(axis=-1) / jnp.maximum(eps, acc)
        # expectation = lambda x: jnp.where(weights==weights.max(axis=-1)).max(axis=-1)#sum(axis=-1) #/ jnp.maximum(eps, acc)

        expectation_3c = lambda x: (weights[..., None] * x).sum(axis=-2) / eps  # jnp.maximum(eps, acc[..., None]
        if trans_atten.shape[-1] == 3:
            rendering['E_map'] = jax.lax.stop_gradient(expectation_3c(trans_atten))
        else:
            rendering['E_map'] = jax.lax.stop_gradient(expectation(trans_atten[..., 0]))

        # jax.debug.print("{y}", y=t_mids[:, 0])
        # rendering['distance_check'] = jax.lax.stop_gradient(expectation(t_mids))
        # For numerical stability this expectation is computing using log-distance.
        rendering['distance_mean'] = (
            jnp.clip(
                jnp.nan_to_num(jnp.exp(expectation(jnp.log(t_mids))), jnp.inf),
                tdist[..., 0], tdist[..., -1]))

        # rendering['distance_mean'] = (
        #   jnp.clip(
        #     jnp.nan_to_num((expectation((t_mids))), jnp.inf),
        #     tdist[..., 0], jnp.inf))
        # idx = jnp.where(weights==weights.max(axis=-1),size=weights.max(axis=-1).size)
        # idx = jnp.argmax(weights,axis=-1)
        # rendering['distance_mean']=jnp.take(t_mids,idx)   #.at[idx].set(tdist[idx])

        # Add an extra fencepost with the far distance at the end of each ray, with
        # whatever weight is needed to make the new weight vector sum to exactly 1
        # (`weights` is only guaranteed to sum to <= 1, not == 1).
        t_aug = jnp.concatenate([tdist, t_far], axis=-1)
        weights_aug = jnp.concatenate([weights, bg_w], axis=-1)

        ps = [5, 50, 95]
        distance_percentiles = stepfun.weighted_percentile(t_aug, weights_aug, ps)
        if trans_atten.shape[-1] == 3:
            emap_percentiles_r = stepfun.weighted_percentile(trans_atten[:, :, 0], weights[:, :-1], ps)[:, None, :]
            emap_percentiles_g = stepfun.weighted_percentile(trans_atten[:, :, 1], weights[:, :-1], ps)[:, None, :]
            emap_percentiles_b = stepfun.weighted_percentile(trans_atten[:, :, 2], weights[:, :-1], ps)[:, None, :]
            emap_percentiles = jnp.concatenate([emap_percentiles_r, emap_percentiles_b, emap_percentiles_g], axis=1)
        else:
            emap_percentiles = stepfun.weighted_percentile(trans_atten[..., 0], weights[:, :-1], ps)

        for i, p in enumerate(ps):
            s = 'median' if p == 50 else  'percentile_' + str(p)
            rendering['distance_' + s] = distance_percentiles[..., i]
            if trans_atten.shape[-1] == 3:
                rendering['E_map_' + s] = emap_percentiles[:, :, i]
            else:
                rendering['E_map_' + s] = emap_percentiles[..., i]

    return rendering


def volumetric_rendering_uw_simon(density, rgbs, c_med, alpha_bs,alpha_atten,
                                  trans_bs,trans_atten, trans, weights,
                                  tdist,
                                  sigma_bs,sigma_atten,
                                  t_far,
                                  compute_extras,
                                  extras=None):
    """Volumetric Rendering Function.

    Args:
      rgbs: jnp.ndarray(float32), color, [batch_size, num_samples, 3]
      weights: jnp.ndarray(float32), weights, [batch_size, num_samples].
      tdist: jnp.ndarray(float32), [batch_size, num_samples].
      bg_rgbs: jnp.ndarray(float32), the color(s) to use for the background.
      t_far: jnp.ndarray(float32), [batch_size, 1], the distance of the far plane.
      compute_extras: bool, if True, compute extra quantities besides color.
      extras: dict, a set of values along rays to render by alpha compositing.

    Returns:
      rendering: a dict containing an rgb image of size [batch_size, 3], and other
        visualizations if compute_extras=True.
    """
    eps = jnp.finfo(jnp.float32).eps
    rendering = {}

    acc = weights.sum(axis=-1)
    bg_w = jnp.maximum(0, 1 - acc[..., None])  # The weight of the background.
    # rgb = (weights[..., None] * rgbs).sum(axis=-2) + bg_w * bg_rgbs
    J = (weights[..., None] * rgbs).sum(axis=-2)
    # direct = ((density[..., None] * rgbs * trans[..., None] * trans_bs * alpha_bs) / (density + sigma_bs)).sum(axis=-2)
    direct = ((density[..., None] * rgbs * trans[..., None] * trans_atten * alpha_atten) / (density[..., None] + sigma_atten[..., None, :])).sum(axis=-2)

    bs = ((sigma_bs[..., None, :] * c_med[..., None, :] * trans[..., None] * trans_bs * alpha_bs) / (density[..., None] + sigma_bs[..., None, :])).sum(
        axis=-2)  # TODO Naama changes, broadcast for c_med
    rgb = direct + bs  # J
    acc = weights.sum(axis=-1)

    rendering['rgb'] = rgb
    rendering['bs'] = bs
    rendering['density'] = density
    rendering['J'] = J
    rendering['direct'] = direct
    rendering['c_med'] = c_med
    rendering['E_map'] = jax.lax.stop_gradient((weights[..., None] * trans_bs).sum(axis=-2))
    rendering['weights'] = weights

    if compute_extras:
        rendering['acc'] = acc

        if extras is not None:
            for k, v in extras.items():
                if v is not None:
                    rendering[k] = (weights[..., None] * v).sum(axis=-2)

        expectation = lambda x: ((trans[..., None] * trans_bs * alpha_bs)[..., 1] * x).sum(axis=-1) / jnp.maximum(eps,
                                                                                                                  acc)
        t_mids = 0.5 * (tdist[..., :-1] + tdist[..., 1:])
        # rendering['distance_check'] = jax.lax.stop_gradient(expectation(t_mids))
        # For numerical stability this expectation is computing using log-distance.
        rendering['distance_mean'] = (
            jnp.clip(
                jnp.nan_to_num(jnp.exp(expectation(jnp.log(t_mids))), jnp.inf),
                tdist[..., 0], tdist[..., -1]))

        # Add an extra fencepost with the far distance at the end of each ray, with
        # whatever weight is needed to make the new weight vector sum to exactly 1
        # (`weights` is only guaranteed to sum to <= 1, not == 1).
        t_aug = jnp.concatenate([tdist, t_far], axis=-1)
        weights_aug = jnp.concatenate([weights, bg_w], axis=-1)

        ps = [5, 50, 95]
        distance_percentiles = stepfun.weighted_percentile(t_aug, weights_aug, ps)

        for i, p in enumerate(ps):
            s = 'median' if p == 50 else 'percentile_' + str(p)
            rendering['distance_' + s] = distance_percentiles[..., i]

    return rendering


def volumetric_rendering(rgbs,
                         weights,
                         tdist,
                         bg_rgbs,
                         t_far,
                         compute_extras,
                         extras=None):
    """Volumetric Rendering Function.

    Args:
      rgbs: jnp.ndarray(float32), color, [batch_size, num_samples, 3]
      weights: jnp.ndarray(float32), weights, [batch_size, num_samples].
      tdist: jnp.ndarray(float32), [batch_size, num_samples].
      bg_rgbs: jnp.ndarray(float32), the color(s) to use for the background.
      t_far: jnp.ndarray(float32), [batch_size, 1], the distance of the far plane.
      compute_extras: bool, if True, compute extra quantities besides color.
      extras: dict, a set of values along rays to render by alpha compositing.

    Returns:
      rendering: a dict containing an rgb image of size [batch_size, 3], and other
        visualizations if compute_extras=True.
    """
    eps = jnp.finfo(jnp.float32).eps
    rendering = {}

    acc = weights.sum(axis=-1)
    bg_w = jnp.maximum(0, 1 - acc[..., None])  # The weight of the background.
    rgb = (weights[..., None] * rgbs).sum(axis=-2) + bg_w * bg_rgbs
    rendering['rgb'] = rgb
    rendering['t_dist'] = tdist

    if compute_extras:
        rendering['acc'] = acc

        if extras is not None:
            for k, v in extras.items():
                if v is not None:
                    rendering[k] = (weights[..., None] * v).sum(axis=-2)

        expectation = lambda x: (weights * x).sum(axis=-1) / jnp.maximum(eps, acc)
        t_mids = 0.5 * (tdist[..., :-1] + tdist[..., 1:])
        # For numerical stability this expectation is computing using log-distance.
        rendering['distance_mean'] = (
            jnp.clip(
                jnp.nan_to_num(jnp.exp(expectation(jnp.log(t_mids))), jnp.inf),
                tdist[..., 0], tdist[..., -1]))

        # Add an extra fencepost with the far distance at the end of each ray, with
        # whatever weight is needed to make the new weight vector sum to exactly 1
        # (`weights` is only guaranteed to sum to <= 1, not == 1).
        t_aug = jnp.concatenate([tdist, t_far], axis=-1)
        weights_aug = jnp.concatenate([weights, bg_w], axis=-1)

        ps = [5, 50, 95]
        distance_percentiles = stepfun.weighted_percentile(t_aug, weights_aug, ps)

        for i, p in enumerate(ps):
            s = 'median' if p == 50 else 'percentile_' + str(p)
            rendering['distance_' + s] = distance_percentiles[..., i]

    return rendering
