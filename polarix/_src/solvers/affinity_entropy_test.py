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

import functools
import itertools

from absl.testing import absltest
from absl.testing import parameterized
import distrax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from polarix._src.games import test_utils
from polarix._src.solvers import affinity_entropy
from polarix._src.solvers import run


EXPECTED_MARGINAL = np.asarray([1.0 / 6.0, 1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0])
EXPECTED_MARGINAL_CLONE_MASKED = np.asarray(
    [0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
)
EXPECTED_MARGINAL_UNIQUE_MASKED = np.asarray(
    [1.0 / 4.0, 1.0 / 4.0, 1.0 / 2.0, 0.0]
)

EXPECTED_JOINT = np.outer(EXPECTED_MARGINAL, EXPECTED_MARGINAL)
EXPECTED_JOINT_CLONE_MASKED = np.outer(
    EXPECTED_MARGINAL_CLONE_MASKED, EXPECTED_MARGINAL_CLONE_MASKED
)
EXPECTED_JOINT_UNIQUE_MASKED = np.outer(
    EXPECTED_MARGINAL_UNIQUE_MASKED, EXPECTED_MARGINAL_UNIQUE_MASKED
)

MASK_CLONE = np.array([False, True, True, True])
MASK_UNIQUE = np.array([True, True, True, False])

# Test cases.
PROBS = [0.0, 0.5, 1.0]
MASKS = (np.ones(4), MASK_CLONE, MASK_UNIQUE)
EXPECTED_MARGINALS = (
    EXPECTED_MARGINAL,
    EXPECTED_MARGINAL_CLONE_MASKED,
    EXPECTED_MARGINAL_UNIQUE_MASKED,
)
EXPECTED_JOINTS = (
    EXPECTED_JOINT,
    EXPECTED_JOINT_CLONE_MASKED,
    EXPECTED_JOINT_UNIQUE_MASKED,
)


class AffinityEntropyTest(parameterized.TestCase):

  def test_identity_kernel_entropy(self):
    key = jax.random.PRNGKey(42)
    dist = distrax.Categorical(jax.random.uniform(key, shape=(4,)))

    identity_kernels = affinity_entropy.identity_kernel()(test_utils.RRPS)
    mask = np.ones(4)
    entropy = affinity_entropy.affinity_entropy(
        dist.probs, identity_kernels[0], mask
    )

    np.testing.assert_allclose(entropy, dist.entropy(), atol=1e-6)

  def test_affinity_kernel_entropy(self):
    key = jax.random.PRNGKey(42)
    dist = distrax.Categorical(jnp.ones(4))

    affinity_kernels = affinity_entropy.affinity_kernel(key=key)(
        test_utils.RRPS
    )
    mask = np.ones(4)
    entropy = affinity_entropy.affinity_entropy(
        dist.probs, affinity_kernels[0], mask
    )
    np.testing.assert_allclose(entropy, 1.039721, atol=1e-6)

  @parameterized.parameters(
      itertools.product(PROBS, zip(MASKS, EXPECTED_MARGINALS))
  )
  def test_rrps_marginals(self, p: float, mask_and_expected):
    mask, expected = mask_and_expected
    game = test_utils.make_rrps()
    key = jax.random.PRNGKey(42)
    masks = [mask, mask]
    solver = functools.partial(
        affinity_entropy.max_affinity_entropy_marginals,
        kernel_fn=affinity_entropy.affinity_kernel(key),
        masks=masks,
        optim=optax.adam(1e-2),
        p=p,
        early_stopping=1_000,
    )
    res = run.solve(
        game, solver, num_iterations_per_update=1_000, max_num_iterations=10_000
    )
    for marginal_p in res.marginals:
      np.testing.assert_allclose(marginal_p, expected, atol=1e-6, rtol=1e-6)

  @parameterized.parameters(
      itertools.product(PROBS, zip(MASKS, EXPECTED_JOINTS))
  )
  def test_rrps_joint(self, p: float, mask_and_expected):
    mask, expected = mask_and_expected
    game = test_utils.make_rrps()
    key = jax.random.PRNGKey(42)
    masks = [mask, mask]
    solver = functools.partial(
        affinity_entropy.max_affinity_entropy_joint,
        kernel_fn=affinity_entropy.affinity_kernel(key),
        masks=masks,
        optim=optax.adam(1e-2),
        p=p,
        early_stopping=1_000,
    )
    res = run.solve(
        game, solver, num_iterations_per_update=1_000, max_num_iterations=10_000
    )
    np.testing.assert_allclose(
        res.joint.flatten(), expected.flatten(), atol=1e-6, rtol=1e-6
    )


if __name__ == '__main__':
  absltest.main()
