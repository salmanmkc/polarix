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

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from polarix._src.solvers import residual


class ResidualTest(parameterized.TestCase):

  @parameterized.product(
      seed=[1, 2, 3],
      shape=[(100,)],
      flip_sign=[True, False],
  )
  def test_residual_non_negative(self, seed, shape, flip_sign):
    max_perturb = 1e-1
    key = jax.random.key(seed)
    params = jnp.zeros(shape)  # Start on boundary.
    for _ in range(10):
      key, grads_key = jax.random.split(key)
      grads = jax.random.uniform(
          grads_key, shape=shape, minval=-max_perturb, maxval=max_perturb
      )
      residuals = residual.residual_non_negative(
          params, grads, flip_sign=flip_sign
      )
      params = jax.tree_util.tree_map(
          lambda p, r: p - r if flip_sign else p + r, params, residuals
      )
    self.assertTrue(np.all(params >= -max_perturb))

  @parameterized.product(
      seed=[1, 2, 3],
      shape=[(100,)],
      flip_sign=[True, False],
  )
  def test_residual_box(self, seed, shape, flip_sign):
    max_perturb = 1e-1
    key = jax.random.key(seed)
    params = jnp.zeros(shape)  # Start on boundary.
    lower = jnp.asarray(0.0)
    upper = jnp.asarray(1.0)
    for _ in range(100):
      key, grads_key = jax.random.split(key)
      grads = jax.random.uniform(
          grads_key, shape=shape, minval=-max_perturb, maxval=max_perturb
      )
      residuals = residual.residual_box(
          params, grads, lower=lower, upper=upper, flip_sign=flip_sign
      )
      params = jax.tree_util.tree_map(
          lambda p, r: p - r if flip_sign else p + r, params, residuals
      )
    self.assertTrue(np.all(params >= -max_perturb))
    self.assertTrue(np.all(params <= 1 + max_perturb))

  @parameterized.product(
      seed=[1, 2, 3],
      size=[100],
      flip_sign=[True, False],
      one_hot=[True, False],
  )
  def test_residual_simplex(self, seed, size, flip_sign, one_hot):
    max_perturb = 1e-2
    key = jax.random.key(seed)
    if one_hot:
      params = jax.nn.one_hot(0, size)  # Start on boundary.
    else:
      params = jnp.ones([size])  # Start on boundary.
      params /= jnp.sum(params)
    for _ in range(1000):
      key, grads_key = jax.random.split(key)
      grads = jax.random.uniform(
          grads_key, shape=[size], minval=-max_perturb, maxval=max_perturb
      )
      residuals = residual.residual_simplex(params, grads, flip_sign=flip_sign)
      params = jax.tree_util.tree_map(
          lambda p, r: p - r if flip_sign else p + r, params, residuals
      )
    self.assertTrue(np.all(params >= -2 * max_perturb))
    self.assertTrue(np.all(params <= 1 + 2 * max_perturb))
    self.assertAlmostEqual(1.0, np.sum(params), places=5)


if __name__ == "__main__":
  absltest.main()
