# Copyright 2025 The polarix Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Additional optimizer functions."""

import functools

import chex
import jax
import jax.numpy as jnp


def _residual_non_negative(
    param: chex.Array, grad: chex.Array, flip_sign: bool = True
) -> chex.ArrayTree:
  if flip_sign:
    return ((param > 0.0) | (grad < 0.0)) * grad
  else:
    return ((param > 0.0) | (grad > 0.0)) * grad


def residual_non_negative(
    params: chex.ArrayTree, grads: chex.ArrayTree, flip_sign: bool = True
) -> chex.ArrayTree:
  """Returns non-negative residuals.

  Residuals are the gradients after removing components that would take the
  parameters outside of the feasible region.

  Assumes params have already been projected onto their nonnegative orthant.

  Args:
    params: The parameters.
    grads: The gradients with respect to the parameters.
    flip_sign: Whether to flip the sign of the gradients. If performing gradient
      descent on a loss function this should be True.

  Returns:
    residuals: Tree of residuals.
  """
  return jax.tree_util.tree_map(
      functools.partial(_residual_non_negative, flip_sign=flip_sign),
      params,
      grads,
  )


def _residual_box(
    param: chex.Array,
    grad: chex.Array,
    lower: chex.Array | None,
    upper: chex.Array | None,
    flip_sign: bool = True,
) -> chex.ArrayTree:
  """Residual box."""
  is_not_lower_param = param > lower
  is_not_upper_param = param < upper
  is_pos_grad = grad > 0.0
  is_neg_grad = grad < 0.0

  mask = True

  if flip_sign:
    if lower is not None:
      mask *= is_not_lower_param | is_neg_grad
    if upper is not None:
      mask *= is_not_upper_param | is_pos_grad
  else:
    if lower is not None:
      mask *= is_not_lower_param | is_pos_grad
    if upper is not None:
      mask *= is_not_upper_param | is_neg_grad

  return mask * grad


def residual_box(
    params: chex.ArrayTree,
    grads: chex.ArrayTree,
    lower: chex.ArrayTree,
    upper: chex.ArrayTree,
    flip_sign: bool = True,
) -> chex.ArrayTree:
  """Returns box residuals.

  Residuals are the gradients after removing components that would take the
  parameters outside of the feasible region.

  Assumes params have already been projected onto the box.

  Args:
    params: Tree of parameters.
    grads: Tree of gradients with respect to the parameters.
    lower: Tree of lower bounds.
    upper: Tree of upper bounds.
    flip_sign: Whether to flip the sign of the gradients. If performing gradient
      descent on a loss function this should be True.

  Returns:
    residuals: Tree of residuals.
  """
  return jax.tree_util.tree_map(
      functools.partial(_residual_box, flip_sign=flip_sign),
      params,
      grads,
      lower,
      upper,
  )


def _residual_simplex(
    param: chex.Array, grad: chex.Array, flip_sign: bool = True
) -> chex.ArrayTree:
  if flip_sign:
    mask = (param > 0.0) | (grad < 0.0)
  else:
    mask = (param > 0.0) | (grad > 0.0)
  residual = mask * grad
  residual = jnp.where(mask, residual - jnp.mean(residual, where=mask), 0.0)
  return residual


def residual_simplex(
    params: chex.ArrayTree, grads: chex.ArrayTree, flip_sign: bool = True
) -> chex.ArrayTree:
  """Returns simplex residuals.

  Residuals are the gradients after removing components that would take the
  parameters outside of the feasible region.

  Assumes params have already been projected onto the simplex.

  Args:
    params: Tree of parameters.
    grads: Tree of gradients with respect to the parameters.
    flip_sign: Whether to flip the sign of the gradients. If performing gradient
      descent on a loss function this should be True.

  Returns:
    residuals: Tree of residuals.
  """
  return jax.tree_util.tree_map(
      functools.partial(_residual_simplex, flip_sign=flip_sign),
      params,
      grads,
  )
