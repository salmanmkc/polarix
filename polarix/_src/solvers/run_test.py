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

import contextlib
import io

from absl.testing import absltest
import numpy as np
from polarix._src.games import base
from polarix._src.solvers import run


class RunTest(absltest.TestCase):

  def test_ne_result(self):

    with self.subTest("raise_on_2d_marginals"):
      with self.assertRaisesRegex(
          AssertionError, ".*assert_trees_all_equal_sizes.*"
      ):
        run.MarginalRatings(
            ratings=[np.zeros((3)), np.zeros((2))],
            marginals=[np.zeros((3, 2))],
        )

    with self.subTest("raise_on_num_actions_mismatch"):
      with self.assertRaisesRegex(AssertionError, ".*size.*"):
        run.MarginalRatings(
            ratings=[np.zeros((3)), np.zeros((2))],
            marginals=[np.zeros((3)), np.zeros((4))],
        )

  def test_ce_result(self):

    with self.subTest("raise_on_3d_joint"):
      with self.assertRaisesRegex(AssertionError, ".*assert_shape.*"):
        run.JointRatings(
            ratings=[np.zeros((3)), np.zeros((2))],
            joint=np.zeros((3, 2, 1)),
        )

    with self.subTest("raise_on_num_actions_mismatch"):
      with self.assertRaisesRegex(AssertionError, ".*assert_shape.*"):
        run.JointRatings(
            ratings=[np.zeros((3)), np.zeros((2))],
            joint=np.zeros((3, 4)),
        )

  def test_run_not_iterator(self):

    # With return type annotation this would not build.
    def _return_once(game):
      del game
      return run.Ratings(ratings=[np.zeros((3))])

    with self.assertRaisesRegex(TypeError, ".*iterator.*"):
      run.solve(
          game=base.Game(
              payoffs=np.ones((2, 2, 3)),
              actions=(np.arange(2), np.arange(3)),
              players=("p1", "p2"),
          ),
          solver=_return_once,
      )

  def test_run_disabled_progress_bar(self):

    def _yield(game):
      del game
      while True:
        yield run.Ratings(ratings=[np.zeros((3))])

    with contextlib.redirect_stderr(io.StringIO()) as fe_enabled:
      run.solve(
          game=base.Game(
              payoffs=np.ones((2, 2, 3)),
              actions=(np.arange(2), np.arange(3)),
              players=("p1", "p2"),
          ),
          solver=_yield,
          num_iterations_per_update=1,
          max_num_iterations=4,
      )

    with contextlib.redirect_stderr(io.StringIO()) as fe_disabled:
      run.solve(
          game=base.Game(
              payoffs=np.ones((2, 2, 3)),
              actions=(np.arange(2), np.arange(3)),
              players=("p1", "p2"),
          ),
          solver=_yield,
          num_iterations_per_update=1,
          max_num_iterations=4,
          disable_progress_bar=True,
      )

    self.assertNotEmpty(fe_enabled.getvalue())
    self.assertEmpty(fe_disabled.getvalue())


if __name__ == "__main__":
  absltest.main()
