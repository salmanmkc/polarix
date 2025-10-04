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
from polarix._src.games import base
from polarix._src.games import test_utils
from polarix._src.solvers import lle
from polarix._src.solvers import run


class LLETest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self.rnd = np.random.RandomState(12345)

  @parameterized.parameters([
      dict(na=(2, 3, 5, 6), num_iters=10, tau=1e-1),
  ])
  def test_random_pt(self, na, num_iters, tau):
    npl = len(na)
    payoffs = np.random.rand(npl, *na)
    game = base.Game(
        payoffs=payoffs,
        actions=tuple([np.arange(nai) for nai in na]),
        players=("a", "b", "c", "d"),
    )
    res = run.solve(
        game=game,
        solver=functools.partial(
            lle.lle,
            anneal_rate=1.0,
            init_temperature=tau,
        ),
        num_iterations_per_update=10,
        max_num_iterations=num_iters,
    )
    self.assertLen(res.ratings, npl)
    self.assertLen(res.marginals, npl)
    self.assertTrue(np.isfinite(res.extra["loss"]))
    self.assertTrue(np.isfinite(res.extra["trigger"]))
    self.assertTrue(np.isfinite(res.extra["exp"]))
    for p in range(npl):
      self.assertTrue(np.all(np.isfinite(res.ratings[p])))
      self.assertTrue(np.all(np.isfinite(res.marginals[p])))

  @parameterized.parameters([
      dict(
          factory=test_utils.make_rrps,
          expected_ratings=(np.zeros(4), np.zeros(4)),
          expected_marginals=(
              np.array([0.166641, 0.166641, 0.334666, 0.332052]),
              np.array([0.166641, 0.166641, 0.334666, 0.332052]),
          ),
      ),
      dict(
          factory=test_utils.make_dominated,
          expected_ratings=(
              np.array([-2.0, 0.0, -1.0]),
              np.array([-2.0, 0.0, -1.0]),
          ),
          expected_marginals=(
              np.array([0.0, 1.0, 0.0]),
              np.array([0.0, 1.0, 0.0]),
          ),
      ),
      dict(
          factory=functools.partial(test_utils.make_chicken, duplicate=False),
          expected_ratings=(
              np.array([0.000673, -0.00734]),
              np.array([0.000673, -0.00734]),
          ),
          expected_marginals=(
              np.array([0.915994, 0.084006]),
              np.array([0.915994, 0.084006]),
          ),
      ),
      dict(
          factory=functools.partial(test_utils.make_el_farol, n=10, c=0.7),
          expected_ratings=(np.array([0.002054, -0.005089]),) * 10,
          expected_marginals=(np.array([0.712388, 0.287611]),) * 10,
      ),
  ])
  def test_lle(self, factory, expected_ratings, expected_marginals):
    game = factory()
    res = run.solve(
        game=game,
        solver=lle.lle,
        num_iterations_per_update=8_000,
        max_num_iterations=40_000,
    )
    max_rating = max(jax.tree_util.tree_map(jnp.max, res.ratings))

    with self.subTest("exp"):
      self.assertLessEqual(res.extra["exp"], 1e-2)
    with self.subTest("ratings"):
      self.assertLessEqual(max_rating, 1e-2 + 1e-3)

    # Consistency.
    with self.subTest("consistent"):
      np.testing.assert_allclose(res.marginals, expected_marginals, atol=4e-3)
      np.testing.assert_allclose(res.ratings, expected_ratings, atol=4e-3)


if __name__ == "__main__":
  absltest.main()
