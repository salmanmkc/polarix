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

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from polarix._src.games import test_utils
from polarix._src.solvers import ce_maxent
from polarix._src.solvers import run


class MaxentCETest(parameterized.TestCase):

  @parameterized.parameters([
      dict(
          factory=test_utils.make_rrps,
          max_ce_gain=0.0,
          target_logit=None,
          expected_ratings=(np.zeros(4), np.zeros(4)),
          expected_joint=(
              np.array([
                  [0.0278, 0.0278, 0.0556, 0.0556],
                  [0.0278, 0.0278, 0.0556, 0.0556],
                  [0.0556, 0.0556, 0.1111, 0.1111],
                  [0.0556, 0.0556, 0.1111, 0.1111],
              ])
          ),
      ),
      dict(
          factory=test_utils.make_dominated,
          max_ce_gain=0.0,
          target_logit=None,
          expected_ratings=[
              np.array([-1.993448e00, 2.168817e-06, -1.002376e00]),
              np.array([-1.996125e00, 5.961857e-04, -9.994118e-01]),
          ],
          expected_joint=np.array([
              [5.951812e-04, 2.679377e-03, 9.549964e-07],
              [2.605322e-30, 9.967235e-01, 4.951431e-08],
              [6.267293e-16, 1.000007e-06, 0.000000e00],
          ]),
      ),
      dict(
          factory=test_utils.make_chicken,
          max_ce_gain=0.0,
          target_logit=None,
          expected_ratings=np.array([
              [1.966953e-06, -4.178298e00, -4.178298e00],
              [1.966953e-06, -4.178298e00, -4.178298e00],
          ]),
          expected_joint=(
              np.array([
                  [0.17291, 0.197782, 0.197782],
                  [0.197782, 0.00899, 0.00899],
                  [0.197782, 0.00899, 0.00899],
              ])
          ),
      ),
      dict(
          factory=test_utils.make_chicken,
          max_ce_gain=0.0,
          target_logit=np.array([
              [0.17291, 0.197782, 10.197782],
              [0.197782, 0.00899, 0.00899],
              [0.197782, 0.00899, 0.00899],
          ]),
          expected_ratings=np.array([
              [2.000003e-06, -1.099844e01, -1.099844e01],
              [2.000003e-06, -1.099844e01, -1.099844e01],
          ]),
          expected_joint=(
              np.array([
                  [3.906463e-05, 4.539158e-05, 9.998161e-01],
                  [4.539158e-05, 2.108708e-06, 2.108708e-06],
                  [4.539158e-05, 2.108708e-06, 2.108708e-06],
              ])
          ),
      ),
  ])
  def test_ce_maxent(
      self, factory, max_ce_gain, target_logit, expected_ratings, expected_joint
  ):
    game = factory()
    solver = functools.partial(
        ce_maxent.ce_maxent, max_ce_gain=max_ce_gain, target_logit=target_logit
    )
    res = run.solve(
        game=game,
        solver=solver,
        num_iterations_per_update=8_000,
        max_num_iterations=40_000,
    )
    max_rating = max(jax.tree_util.tree_map(jnp.max, res.ratings))

    with self.subTest("ce_gap"):
      self.assertAlmostEqual(res.extra["ce_gap"], max_ce_gain, places=2)
    with self.subTest("residual_norm"):
      self.assertAlmostEqual(res.extra["residual_norm"], 0.0, places=2)
    with self.subTest("ratings"):
      self.assertLessEqual(max_rating, (8 - 1) * max_ce_gain + 1e-3)

    # Consistency.
    with self.subTest("consistent"):
      np.testing.assert_allclose(expected_joint, res.joint, atol=4e-3)
      np.testing.assert_allclose(expected_ratings, res.ratings, atol=4e-3)


if __name__ == "__main__":
  absltest.main()
